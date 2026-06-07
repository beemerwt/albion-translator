use crate::translation::{
    Backend, CachedTranslator, Ct2Config, Ct2Translator, NoopTranslator,
    TranslateRequest as NativeTranslateRequest, TranslationConfig, Translator,
    simple_detect_language,
};
use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{
    env,
    path::PathBuf,
    process::{Child, Command, ExitStatus},
    thread,
    time::{Duration, Instant},
};

const DEFAULT_PORT: u16 = 8787;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(100);
const INITIAL_SOURCE: &str = "es";
const DEFAULT_TARGET: &str = "en";

pub struct TranslatorServer {
    backend: TranslationBackend,
}

impl TranslatorServer {
    pub fn start() -> Result<Self> {
        let config = TranslationConfig::from_env()?;
        let backend = TranslationBackend::start(config)?;
        Ok(Self { backend })
    }

    pub fn wait_for_health(&mut self) -> Result<()> {
        self.backend.wait_for_health()
    }

    pub fn translate_to_english(&self, text: &str) -> Result<TranslateResponse> {
        self.translate_to_english_from(text, "auto")
    }

    pub fn translate_to_english_from(&self, text: &str, source: &str) -> Result<TranslateResponse> {
        self.backend.translate_to_english_from(text, source)
    }

    pub fn detect_language(&self, text: &str) -> Result<DetectResponse> {
        self.backend.detect_language(text)
    }
}

enum TranslationBackend {
    Native(Box<dyn Translator>),
    Argos(ArgosServer),
    Unavailable(String),
}

impl TranslationBackend {
    fn start(config: TranslationConfig) -> Result<Self> {
        match config.backend {
            Backend::Noop => Ok(Self::Native(Box::new(NoopTranslator))),
            Backend::Ct2 => Ok(start_ct2_backend(&config).unwrap_or_else(|error| {
                Self::Unavailable(format!("ct2 translation is unavailable: {error:#}"))
            })),
            Backend::Argos => ArgosServer::start().map(Self::Argos).or_else(|error| {
                Ok(Self::Unavailable(format!(
                    "deprecated Argos sidecar is unavailable: {error:#}"
                )))
            }),
            Backend::Auto => {
                if Ct2Translator::is_available(&ct2_config(&config), INITIAL_SOURCE, DEFAULT_TARGET)
                {
                    if let Ok(backend) = start_ct2_backend(&config) {
                        return Ok(backend);
                    }
                }

                if config.argos_fallback {
                    match ArgosServer::start() {
                        Ok(server) => {
                            eprintln!(
                                "warning: using deprecated Python Argos translation fallback; install a CTranslate2 model to use the Rust-native ct2 backend"
                            );
                            Ok(Self::Argos(server))
                        }
                        Err(error) => Ok(Self::Unavailable(format!(
                            "translation is unavailable; no ct2 model is installed and deprecated Argos fallback failed: {error:#}"
                        ))),
                    }
                } else {
                    Ok(Self::Unavailable(
                        "translation is unavailable; no ct2 model is installed and Argos fallback is disabled"
                            .to_string(),
                    ))
                }
            }
        }
    }

    fn wait_for_health(&mut self) -> Result<()> {
        match self {
            Self::Argos(server) => server.wait_for_health(),
            Self::Native(_) | Self::Unavailable(_) => Ok(()),
        }
    }

    fn translate_to_english_from(&self, text: &str, source: &str) -> Result<TranslateResponse> {
        match self {
            Self::Native(translator) => {
                let source = if source == "auto" {
                    simple_detect_language(text)
                } else {
                    source.to_string()
                };
                let response = translator.translate(NativeTranslateRequest {
                    source_lang: source.clone(),
                    target_lang: DEFAULT_TARGET.to_string(),
                    text: text.to_string(),
                })?;

                Ok(TranslateResponse {
                    source,
                    target: DEFAULT_TARGET.to_string(),
                    translated_text: response.translated_text,
                    engine: response.engine,
                    model_id: response.model_id,
                    device: response.device,
                })
            }
            Self::Argos(server) => server.client.translate_to_english_from(text, source),
            Self::Unavailable(reason) => Err(anyhow!("{reason}")),
        }
    }

    fn detect_language(&self, text: &str) -> Result<DetectResponse> {
        match self {
            Self::Native(_) | Self::Unavailable(_) => Ok(DetectResponse {
                language: simple_detect_language(text),
                confidence: None,
            }),
            Self::Argos(server) => server.client.detect_language(text),
        }
    }
}

fn start_ct2_backend(config: &TranslationConfig) -> Result<TranslationBackend> {
    let translator = Ct2Translator::new(ct2_config(config), INITIAL_SOURCE, DEFAULT_TARGET)?;
    let model_id = translator.model_id();
    eprintln!(
        "translation backend: ct2 model={} device={} cuda_build={}",
        model_id.as_deref().unwrap_or("unknown"),
        translator.device(),
        Ct2Translator::cuda_build_available()
    );

    Ok(TranslationBackend::Native(Box::new(CachedTranslator::new(
        translator,
        model_id,
        config.cache_capacity,
    ))))
}

fn ct2_config(config: &TranslationConfig) -> Ct2Config {
    Ct2Config {
        model_dir: config.model_dir.clone(),
        manifest_path: config.manifest_path.clone(),
        device: config.device,
        allow_device_fallback: config.allow_device_fallback,
    }
}

struct ArgosServer {
    child: Option<Child>,
    client: TranslatorClient,
}

impl ArgosServer {
    fn start() -> Result<Self> {
        let port = configured_port()?;
        let client = TranslatorClient::new(port)?;
        let child = if use_external_server() {
            None
        } else {
            Some(spawn_python_server(port)?)
        };

        let mut server = Self { child, client };
        server.wait_for_health()?;
        Ok(server)
    }

    fn wait_for_health(&mut self) -> Result<()> {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        let mut last_error = None;

        while Instant::now() < deadline {
            if let Some(status) = self.child_status()? {
                return Err(anyhow!(
                    "translator sidecar exited before becoming healthy: {status}"
                ));
            }

            match self.client.health() {
                Ok(()) => return Ok(()),
                Err(error) => last_error = Some(error),
            }

            thread::sleep(HEALTH_POLL_INTERVAL);
        }

        Err(last_error.unwrap_or_else(|| anyhow!("translator health check timed out")))
            .context("translator server did not become healthy")
    }

    fn child_status(&mut self) -> Result<Option<ExitStatus>> {
        match self.child.as_mut() {
            Some(child) => child
                .try_wait()
                .context("failed to inspect translator child status"),
            None => Ok(None),
        }
    }
}

impl Drop for ArgosServer {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };

        match child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranslatorClient {
    http: Client,
    base_url: String,
}

impl TranslatorClient {
    pub fn new(port: u16) -> Result<Self> {
        Ok(Self {
            http: Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .context("failed to build translator HTTP client")?,
            base_url: base_url(port),
        })
    }

    pub fn wait_for_health(&self) -> Result<()> {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        let mut last_error = None;

        while Instant::now() < deadline {
            match self.health() {
                Ok(()) => return Ok(()),
                Err(error) => {
                    last_error = Some(error);
                }
            }

            thread::sleep(HEALTH_POLL_INTERVAL);
        }

        Err(last_error.unwrap_or_else(|| anyhow!("translator health check timed out")))
            .context("translator server did not become healthy")
    }

    fn health(&self) -> Result<()> {
        let response = self
            .http
            .get(self.url("/health"))
            .send()
            .context("failed to send translator health request")?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "translator health check returned HTTP {}",
                response.status()
            ));
        }

        let health: HealthResponse = response
            .json()
            .context("failed to parse translator health response")?;
        if health.ok {
            Ok(())
        } else {
            Err(anyhow!("translator health check returned ok=false"))
        }
    }

    pub fn translate_to_english(&self, text: &str) -> Result<TranslateResponse> {
        self.translate_to_english_from(text, "auto")
    }

    pub fn translate_to_english_from(&self, text: &str, source: &str) -> Result<TranslateResponse> {
        let request = TranslateRequest {
            text,
            source,
            target: DEFAULT_TARGET,
        };

        let mut response: TranslateResponse = self
            .http
            .post(self.url("/translate"))
            .json(&request)
            .send()
            .context("failed to send translation request")?
            .error_for_status()
            .context("translator returned an error status")?
            .json()
            .context("failed to parse translation response")?;
        response.engine = "argos".to_string();
        response.device = "cpu".to_string();
        Ok(response)
    }

    pub fn detect_language(&self, text: &str) -> Result<DetectResponse> {
        let request = DetectRequest { text };

        self.http
            .post(self.url("/detect"))
            .json(&request)
            .send()
            .context("failed to send language detection request")?
            .error_for_status()
            .context("translator returned an error status for language detection")?
            .json()
            .context("failed to parse language detection response")
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    ok: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct DetectRequest<'a> {
    pub text: &'a str,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct DetectResponse {
    pub language: String,
    pub confidence: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct TranslateRequest<'a> {
    pub text: &'a str,
    pub source: &'a str,
    pub target: &'a str,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct TranslateResponse {
    pub source: String,
    pub target: String,
    pub translated_text: String,
    #[serde(default = "default_engine")]
    pub engine: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_engine() -> String {
    "argos".to_string()
}

fn default_device() -> String {
    "cpu".to_string()
}

fn configured_port() -> Result<u16> {
    match env::var("ALBION_TRANSLATOR_PORT") {
        Ok(value) => value
            .parse()
            .with_context(|| format!("invalid ALBION_TRANSLATOR_PORT value {value:?}")),
        Err(env::VarError::NotPresent) => Ok(DEFAULT_PORT),
        Err(error) => Err(error).context("failed to read ALBION_TRANSLATOR_PORT"),
    }
}

fn use_external_server() -> bool {
    matches!(env::var("ALBION_TRANSLATOR_EXTERNAL").as_deref(), Ok("1"))
}

fn spawn_python_server(port: u16) -> Result<Child> {
    let translator_dir = translator_dir()?;
    let python = python_executable(&translator_dir);
    let argos_home = translator_dir.join(".argos");

    Command::new(&python)
        .args([
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .current_dir(&translator_dir)
        .env("ARGOS_DEVICE_TYPE", "cpu")
        .env("XDG_CACHE_HOME", argos_home.join("cache"))
        .env("XDG_CONFIG_HOME", argos_home.join("config"))
        .env("XDG_DATA_HOME", argos_home.join("data"))
        .env("PYTHONUNBUFFERED", "1")
        .spawn()
        .with_context(|| {
            format!(
                "failed to launch deprecated Argos translator sidecar with {} from {}",
                python.display(),
                translator_dir.display()
            )
        })
}

fn translator_dir() -> Result<PathBuf> {
    let local = PathBuf::from("translator");
    if local.is_dir() {
        return local
            .canonicalize()
            .context("failed to resolve translator directory");
    }

    let manifest_relative = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("translator");
    if manifest_relative.is_dir() {
        return manifest_relative
            .canonicalize()
            .context("failed to resolve translator directory");
    }

    Err(anyhow!("translator directory was not found"))
}

fn python_executable(translator_dir: &std::path::Path) -> PathBuf {
    let venv_python = translator_dir.join(".venv").join("bin").join("python");
    if venv_python.is_file() {
        venv_python
    } else {
        PathBuf::from("python3")
    }
}

fn base_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_localhost_base_url() {
        assert_eq!(base_url(8787), "http://127.0.0.1:8787");
    }

    #[test]
    fn falls_back_to_python3_without_venv() {
        assert_eq!(
            python_executable(std::path::Path::new("missing-translator-dir")),
            PathBuf::from("python3")
        );
    }

    #[test]
    fn prefers_venv_python_when_present() {
        let translator_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("translator");
        let expected = translator_dir.join(".venv").join("bin").join("python");

        if expected.is_file() {
            assert_eq!(python_executable(&translator_dir), expected);
        }
    }

    #[test]
    fn serializes_translate_request_contract() {
        let request = TranslateRequest {
            text: "hola",
            source: "auto",
            target: "en",
        };

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "text": "hola",
                "source": "auto",
                "target": "en"
            })
        );
    }

    #[test]
    fn deserializes_argos_translate_response_contract_with_defaults() {
        let response: TranslateResponse = serde_json::from_value(serde_json::json!({
            "source": "es",
            "target": "en",
            "translated_text": "[stub translation es->en] hola"
        }))
        .unwrap();

        assert_eq!(response.source, "es");
        assert_eq!(response.target, "en");
        assert_eq!(response.translated_text, "[stub translation es->en] hola");
        assert_eq!(response.engine, "argos");
        assert_eq!(response.model_id, None);
        assert_eq!(response.device, "cpu");
    }

    #[test]
    fn native_noop_backend_uses_existing_facade() {
        let backend = TranslationBackend::start(TranslationConfig {
            backend: Backend::Noop,
            ..Default::default()
        })
        .unwrap();

        let response = backend.translate_to_english_from("hola HO", "es").unwrap();

        assert_eq!(response.translated_text, "hola HO");
        assert_eq!(response.engine, "noop");
    }

    #[test]
    #[ignore = "starts the deprecated Python Argos sidecar and requires sidecar dependencies"]
    fn starts_sidecar_and_detects_language_over_http() {
        let server = ArgosServer::start().unwrap();
        let response = server.client.detect_language("hola mundo gracias").unwrap();

        assert!(
            ["es", "pt", "zh", "vi", "en", "unknown"].contains(&response.language.as_str()),
            "unexpected detected language: {:?}",
            response
        );
    }

    #[test]
    #[ignore = "starts the deprecated Python Argos sidecar and requires an installed es->en Argos package"]
    fn starts_sidecar_and_translates_spanish_over_http() {
        let server = ArgosServer::start().unwrap();
        let response = server
            .client
            .translate_to_english_from("te gustaria un poco de chocolate", "es")
            .unwrap();

        assert_eq!(response.source, "es");
        assert_eq!(response.target, "en");
        assert!(!response.translated_text.trim().is_empty());
        assert!(!response.translated_text.contains("mainstre"));
        assert!(response.translated_text.chars().count() <= 120);
    }
}
