use crate::translation::{
    Backend, Ct2Config, HttpTemplateBackend, HttpTemplateConfig, LanguageDetection,
    LanguageDetector, LinguaLanguageDetector, SourceLanguageConfig, TranslationCache,
    TranslationConfig, TranslationOutcome, TranslationRouter,
    translator::{
        TranslateRequest as BackendTranslateRequest, TranslateResponse as BackendTranslateResponse,
        Translator,
    },
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
const DEFAULT_TARGET: &str = "en";

pub struct TranslatorServer {
    backend: TranslationBackend,
    cache: TranslationCache,
    target_lang: String,
}

impl TranslatorServer {
    pub fn start() -> Result<Self> {
        let config = TranslationConfig::from_env()?;
        let cache = TranslationCache::open(&config.cache_db_path)?;
        let target_lang = config.target_lang.clone();
        let backend = TranslationBackend::start(config)?;
        Ok(Self {
            backend,
            cache,
            target_lang,
        })
    }

    pub fn wait_for_health(&mut self) -> Result<()> {
        self.backend.wait_for_health()
    }

    pub fn translate_to_english(&self, text: &str) -> Result<TranslateResponse> {
        self.translate_to_english_from(text, "auto")
    }

    pub fn translate_to_english_from(&self, text: &str, source: &str) -> Result<TranslateResponse> {
        if let Some(cached) = self.cache.lookup_google(text, &self.target_lang)? {
            return Ok(cached);
        }

        let response = self.backend.translate_to_english_from(text, source)?;
        if response.engine == "google" {
            self.cache.insert_google(text, &response)?;
        }
        Ok(response)
    }

    pub fn detect_language(&self, text: &str) -> Result<DetectResponse> {
        self.backend.detect_language(text)
    }

    pub fn uses_async_remote_backend(&self) -> bool {
        self.backend.uses_async_remote_backend()
    }
}

enum TranslationBackend {
    Native(Box<TranslationRouter>),
    Argos {
        server: ArgosServer,
        source_lang: SourceLanguageConfig,
        target_lang: String,
    },
    Google {
        client: GoogleTranslateClient,
        detector: LinguaLanguageDetector,
        target_lang: String,
        confidence_threshold: f64,
    },
    Http {
        client: HttpTemplateBackend,
        detector: LinguaLanguageDetector,
        source_lang: SourceLanguageConfig,
        target_lang: String,
        confidence_threshold: f64,
    },
    Noop {
        source_lang: SourceLanguageConfig,
        target_lang: String,
    },
    #[cfg(test)]
    Counting {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        response: TranslateResponse,
    },
    Unavailable(String),
}

impl TranslationBackend {
    fn start(config: TranslationConfig) -> Result<Self> {
        match config.backend {
            Backend::Noop => Ok(Self::Noop {
                source_lang: config.source_lang,
                target_lang: config.target_lang,
            }),
            Backend::Ct2 => Ok(start_native_router(&config).unwrap_or_else(|error| {
                Self::Unavailable(format!("ct2 translation is unavailable: {error:#}"))
            })),
            Backend::Argos => ArgosServer::start()
                .map(|server| Self::Argos {
                    server,
                    source_lang: config.source_lang,
                    target_lang: config.target_lang,
                })
                .or_else(|error| {
                    Ok(Self::Unavailable(format!(
                        "deprecated Argos sidecar is unavailable: {error:#}"
                    )))
                }),
            Backend::Google => Ok(Self::Google {
                client: GoogleTranslateClient::from_env()?,
                detector: LinguaLanguageDetector::default(),
                target_lang: config.target_lang,
                confidence_threshold: config.detection_confidence_threshold,
            }),
            Backend::Http => {
                let path = config.http_config_path.clone().ok_or_else(|| {
                    anyhow!("TRANSLATION_BACKEND=http requires TRANSLATION_HTTP_CONFIG")
                })?;
                Ok(Self::Http {
                    client: HttpTemplateBackend::from_path(path, config.http_api_key.clone())?,
                    detector: LinguaLanguageDetector::default(),
                    source_lang: config.source_lang,
                    target_lang: config.target_lang,
                    confidence_threshold: config.detection_confidence_threshold,
                })
            }
            Backend::TranslateGemmaVllm => Ok(Self::Http {
                client: HttpTemplateBackend::new(
                    match config.http_config_path.clone() {
                        Some(path) => HttpTemplateConfig::from_path(path)?,
                        None => HttpTemplateConfig::translategemma_vllm_preset()?,
                    },
                    config.http_api_key.clone(),
                )?,
                detector: LinguaLanguageDetector::default(),
                source_lang: config.source_lang,
                target_lang: config.target_lang,
                confidence_threshold: config.detection_confidence_threshold,
            }),
            Backend::Auto => {
                match start_native_router(&config) {
                    Ok(Self::Native(router)) if router.has_installed_model_for_target() => {
                        return Ok(Self::Native(router));
                    }
                    Ok(_) => {}
                    Err(error) => {
                        eprintln!(
                            "warning: native ct2 translation router is unavailable: {error:#}"
                        );
                    }
                }

                if config.argos_fallback {
                    match ArgosServer::start() {
                        Ok(server) => {
                            eprintln!(
                                "warning: using deprecated Python Argos translation fallback; install a CTranslate2 model to use the Rust-native ct2 backend"
                            );
                            Ok(Self::Argos {
                                server,
                                source_lang: config.source_lang,
                                target_lang: config.target_lang,
                            })
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
            Self::Argos { server, .. } => server.wait_for_health(),
            Self::Native(_)
            | Self::Google { .. }
            | Self::Http { .. }
            | Self::Noop { .. }
            | Self::Unavailable(_) => Ok(()),
            #[cfg(test)]
            Self::Counting { .. } => Ok(()),
        }
    }

    fn uses_async_remote_backend(&self) -> bool {
        matches!(self, Self::Google { .. } | Self::Http { .. })
    }

    fn translate_to_english_from(&self, text: &str, source: &str) -> Result<TranslateResponse> {
        match self {
            Self::Native(router) => match router.translate_with_source(text, source)? {
                TranslationOutcome::Translated {
                    response,
                    source_lang,
                    target_lang,
                } => Ok(TranslateResponse {
                    source: response
                        .detected_language
                        .clone()
                        .unwrap_or_else(|| source_lang.clone()),
                    target: target_lang,
                    translated_text: response.translated_text,
                    engine: response.engine,
                    model_id: response.model_id,
                    device: response.device,
                    detected_language: response.detected_language.or(Some(source_lang)),
                    from_cache: response.from_cache,
                }),
                TranslationOutcome::Skipped {
                    source_lang,
                    target_lang,
                    ..
                } => Ok(TranslateResponse {
                    source: source_lang.clone(),
                    target: target_lang,
                    translated_text: text.to_string(),
                    engine: "skipped".to_string(),
                    model_id: None,
                    device: "none".to_string(),
                    detected_language: Some(source_lang),
                    from_cache: false,
                }),
            },
            Self::Argos {
                server,
                source_lang,
                target_lang,
            } => {
                server
                    .client
                    .translate(text, &source_for_request(source, source_lang), target_lang)
            }
            Self::Google {
                client,
                detector,
                target_lang,
                confidence_threshold,
            } => {
                let detection = detect_locally_for_routing(detector, text)?;
                if let Some(response) = skip_from_local_detection(
                    text,
                    target_lang,
                    confidence_threshold,
                    detection.as_ref(),
                ) {
                    return Ok(response);
                }

                let local_detected_language = detection
                    .as_ref()
                    .map(|detection| detection.language.as_str());
                let mut response = client.translate(text, target_lang)?;
                apply_google_language_fallback(&mut response, local_detected_language);
                Ok(response)
            }
            Self::Http {
                client,
                detector,
                source_lang,
                target_lang,
                confidence_threshold,
            } => {
                let source = resolve_http_source(detector, source_lang, text, source)?;
                if let Some(response) = skip_from_local_detection(
                    text,
                    target_lang,
                    confidence_threshold,
                    Some(&source),
                ) {
                    return Ok(response);
                }

                let response = client.translate(BackendTranslateRequest {
                    source_lang: source.language.clone(),
                    target_lang: target_lang.clone(),
                    text: text.to_string(),
                })?;
                Ok(http_translate_response(
                    response,
                    source.language,
                    target_lang.clone(),
                ))
            }
            Self::Noop {
                source_lang,
                target_lang,
            } => Ok(TranslateResponse {
                source: source_for_request(source, source_lang),
                target: target_lang.clone(),
                translated_text: text.to_string(),
                engine: "noop".to_string(),
                model_id: None,
                device: "none".to_string(),
                detected_language: Some(source_for_request(source, source_lang)),
                from_cache: false,
            }),
            #[cfg(test)]
            Self::Counting { calls, response } => {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(response.clone())
            }
            Self::Unavailable(reason) => Err(anyhow!("{reason}")),
        }
    }

    fn detect_language(&self, text: &str) -> Result<DetectResponse> {
        match self {
            Self::Native(router) => detect_response(router.detect_language(text)?),
            Self::Google { detector, .. } => detect_response(detector.detect(text)?),
            Self::Http { detector, .. } => detect_response(detector.detect(text)?),
            Self::Noop { .. } | Self::Unavailable(_) => {
                detect_response(LinguaLanguageDetector::default().detect(text)?)
            }
            #[cfg(test)]
            Self::Counting { .. } => {
                detect_response(LinguaLanguageDetector::default().detect(text)?)
            }
            Self::Argos { server, .. } => server.client.detect_language(text),
        }
    }
}

fn resolve_http_source(
    detector: &LinguaLanguageDetector,
    config_source: &SourceLanguageConfig,
    text: &str,
    source: &str,
) -> Result<LanguageDetection> {
    if source != "auto" {
        return Ok(LanguageDetection {
            language: source.to_string(),
            confidence: Some(1.0),
            reliable: true,
        });
    }

    match config_source {
        SourceLanguageConfig::Manual(language) => Ok(LanguageDetection {
            language: language.clone(),
            confidence: Some(1.0),
            reliable: true,
        }),
        SourceLanguageConfig::Auto => detector.detect(text),
    }
}

fn http_translate_response(
    response: BackendTranslateResponse,
    source: String,
    target: String,
) -> TranslateResponse {
    TranslateResponse {
        source,
        target,
        translated_text: response.translated_text,
        engine: response.engine,
        model_id: response.model_id,
        device: response.device,
        detected_language: response.detected_language,
        from_cache: response.from_cache,
    }
}

fn start_native_router(config: &TranslationConfig) -> Result<TranslationBackend> {
    let router = TranslationRouter::new_ct2(
        ct2_config(config),
        config.source_lang.clone(),
        config.target_lang.clone(),
        config.detection_confidence_threshold,
        config.cache_capacity,
    )?;
    eprintln!(
        "translation backend: ct2 router target={} source={:?}",
        config.target_lang, config.source_lang
    );

    Ok(TranslationBackend::Native(Box::new(router)))
}

fn ct2_config(config: &TranslationConfig) -> Ct2Config {
    Ct2Config {
        model_dir: config.model_dir.clone(),
        manifest_path: config.manifest_path.clone(),
        device: config.device,
        allow_device_fallback: config.allow_device_fallback,
    }
}

fn detect_response(detection: LanguageDetection) -> Result<DetectResponse> {
    Ok(DetectResponse {
        language: detection.language,
        confidence: detection.confidence,
    })
}

fn detect_locally_for_routing(
    detector: &LinguaLanguageDetector,
    text: &str,
) -> Result<Option<LanguageDetection>> {
    if skip_without_detection(text).is_some() {
        return Ok(None);
    }

    Ok(Some(detector.detect(text)?))
}

fn skip_from_local_detection(
    text: &str,
    target_lang: &str,
    confidence_threshold: &f64,
    detection: Option<&LanguageDetection>,
) -> Option<TranslateResponse> {
    if skip_without_detection(text).is_some() {
        return Some(skipped_response(text, "unknown", target_lang));
    }

    let detection = detection?;
    let detected_language = detection.language.as_str();
    let confidence = detection.confidence.unwrap_or(0.0);
    if detected_language == "unknown" || confidence < *confidence_threshold {
        return Some(skipped_response(text, detected_language, target_lang));
    }

    if detected_language == target_lang {
        return Some(skipped_response(text, detected_language, target_lang));
    }

    None
}

fn skipped_response(text: &str, source: &str, target: &str) -> TranslateResponse {
    TranslateResponse {
        source: source.to_string(),
        target: target.to_string(),
        translated_text: text.to_string(),
        engine: "skipped".to_string(),
        model_id: None,
        device: "none".to_string(),
        detected_language: Some(source.to_string()),
        from_cache: false,
    }
}

fn apply_google_language_fallback(
    response: &mut TranslateResponse,
    local_detected_language: Option<&str>,
) {
    if response.detected_language.is_none() {
        response.detected_language = local_detected_language.map(str::to_string);
    }
    if response.source == "unknown" {
        response.source = response
            .detected_language
            .clone()
            .or_else(|| local_detected_language.map(str::to_string))
            .unwrap_or_else(|| "unknown".to_string());
    }
}

fn skip_without_detection(text: &str) -> Option<()> {
    let trimmed = text.trim();
    if trimmed.is_empty() || contains_url(trimmed) {
        return Some(());
    }

    let alphabetic_count = trimmed
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    if alphabetic_count < 4 { Some(()) } else { None }
}

fn contains_url(text: &str) -> bool {
    text.split_whitespace().any(|token| {
        let token = token.to_ascii_lowercase();
        token.starts_with("http://")
            || token.starts_with("https://")
            || token.starts_with("www.")
            || token.contains("://")
    })
}

fn source_for_request(source: &str, config_source: &SourceLanguageConfig) -> String {
    if source != "auto" {
        return source.to_string();
    }

    match config_source {
        SourceLanguageConfig::Auto => "auto".to_string(),
        SourceLanguageConfig::Manual(language) => language.clone(),
    }
}

#[derive(Debug, Clone)]
struct GoogleTranslateClient {
    http: Client,
    api_key: String,
    endpoint: String,
}

impl GoogleTranslateClient {
    fn from_env() -> Result<Self> {
        let api_key = read_google_api_key()?;
        let endpoint = env::var("GOOGLE_TRANSLATE_ENDPOINT").unwrap_or_else(|_| {
            "https://translation.googleapis.com/language/translate/v2".to_string()
        });

        Ok(Self {
            http: Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .context("failed to build Google Translate HTTP client")?,
            api_key,
            endpoint,
        })
    }

    fn translate(&self, text: &str, target: &str) -> Result<TranslateResponse> {
        let request = GoogleTranslateRequest {
            q: text,
            target,
            format: "text",
        };

        let response: GoogleTranslateResponse = self
            .http
            .post(&self.endpoint)
            .query(&[("key", self.api_key.as_str())])
            .json(&request)
            .send()
            .context("failed to send Google Translate request")?
            .error_for_status()
            .context("Google Translate returned an error status")?
            .json()
            .context("failed to parse Google Translate response")?;

        let translation = response
            .data
            .translations
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Google Translate returned no translation"))?;
        let detected_language = translation.detected_source_language;
        let source = detected_language
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        Ok(TranslateResponse {
            source,
            target: target.to_string(),
            translated_text: decode_google_text(&translation.translated_text),
            engine: "google".to_string(),
            model_id: None,
            device: "remote".to_string(),
            detected_language,
            from_cache: false,
        })
    }
}

fn read_google_api_key() -> Result<String> {
    match env::var("GOOGLE_TRANSLATE_API_KEY").or_else(|_| env::var("TRANSLATION_GOOGLE_API_KEY")) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) | Err(_) => Err(anyhow!(
            "Google Translate requires GOOGLE_TRANSLATE_API_KEY or TRANSLATION_GOOGLE_API_KEY"
        )),
    }
}

#[derive(Debug, Serialize)]
struct GoogleTranslateRequest<'a> {
    q: &'a str,
    target: &'a str,
    format: &'a str,
}

#[derive(Debug, Deserialize)]
struct GoogleTranslateResponse {
    data: GoogleTranslateData,
}

#[derive(Debug, Deserialize)]
struct GoogleTranslateData {
    translations: Vec<GoogleTranslation>,
}

#[derive(Debug, Deserialize)]
struct GoogleTranslation {
    #[serde(rename = "translatedText")]
    translated_text: String,
    #[serde(rename = "detectedSourceLanguage")]
    detected_source_language: Option<String>,
}

fn decode_google_text(text: &str) -> String {
    text.replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
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
        self.translate(text, source, DEFAULT_TARGET)
    }

    pub fn translate(&self, text: &str, source: &str, target: &str) -> Result<TranslateResponse> {
        let request = TranslateRequest {
            text,
            source,
            target,
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
        response.detected_language = Some(response.source.clone());
        response.from_cache = false;
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    #[serde(default)]
    pub detected_language: Option<String>,
    #[serde(default)]
    pub from_cache: bool,
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
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

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
        assert_eq!(response.detected_language, None);
        assert!(!response.from_cache);
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

    fn google_response(text: &str, detected_language: Option<&str>) -> TranslateResponse {
        TranslateResponse {
            source: detected_language.unwrap_or("unknown").to_string(),
            target: "en".to_string(),
            translated_text: text.to_string(),
            engine: "google".to_string(),
            model_id: None,
            device: "remote".to_string(),
            detected_language: detected_language.map(str::to_string),
            from_cache: false,
        }
    }

    #[test]
    fn cache_hit_avoids_backend_call() {
        let cache = TranslationCache::in_memory().unwrap();
        cache
            .insert_google(
                " hola   amigo ",
                &google_response("hello friend", Some("es")),
            )
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let server = TranslatorServer {
            backend: TranslationBackend::Counting {
                calls: calls.clone(),
                response: google_response("should not be used", Some("es")),
            },
            cache,
            target_lang: "en".to_string(),
        };

        let response = server.translate_to_english("hola amigo").unwrap();

        assert_eq!(response.translated_text, "hello friend");
        assert!(response.from_cache);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cache_miss_calls_backend_and_inserts_google_result() {
        let calls = Arc::new(AtomicUsize::new(0));
        let server = TranslatorServer {
            backend: TranslationBackend::Counting {
                calls: calls.clone(),
                response: google_response("hello friend", Some("es")),
            },
            cache: TranslationCache::in_memory().unwrap(),
            target_lang: "en".to_string(),
        };

        let first = server.translate_to_english("hola amigo").unwrap();
        let second = server.translate_to_english(" hola   amigo ").unwrap();

        assert!(!first.from_cache);
        assert!(second.from_cache);
        assert_eq!(second.translated_text, "hello friend");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn google_detected_language_overrides_local_detection() {
        let mut response = google_response("hello friend", Some("es"));

        apply_google_language_fallback(&mut response, Some("pt"));

        assert_eq!(response.source, "es");
        assert_eq!(response.detected_language.as_deref(), Some("es"));
    }

    #[test]
    fn google_result_falls_back_to_lingua_detection_when_missing() {
        let mut response = google_response("hello friend", None);

        apply_google_language_fallback(&mut response, Some("pt"));

        assert_eq!(response.source, "pt");
        assert_eq!(response.detected_language.as_deref(), Some("pt"));
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
