use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{
    env,
    path::PathBuf,
    process::{Child, Command},
    thread,
    time::{Duration, Instant},
};

const DEFAULT_PORT: u16 = 8787;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(10);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug)]
pub struct TranslatorServer {
    child: Option<Child>,
    client: TranslatorClient,
}

impl TranslatorServer {
    pub fn start() -> Result<Self> {
        let port = configured_port()?;
        let client = TranslatorClient::new(port)?;
        let child = if use_external_server() {
            None
        } else {
            Some(spawn_python_server(port)?)
        };

        let server = Self { child, client };
        server.wait_for_health()?;
        Ok(server)
    }

    pub fn wait_for_health(&self) -> Result<()> {
        self.client.wait_for_health()
    }

    pub fn translate_to_english(&self, text: &str) -> Result<TranslateResponse> {
        self.client.translate_to_english(text)
    }
}

impl Drop for TranslatorServer {
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
                .timeout(Duration::from_secs(2))
                .build()
                .context("failed to build translator HTTP client")?,
            base_url: base_url(port),
        })
    }

    pub fn wait_for_health(&self) -> Result<()> {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        let health_url = self.url("/health");
        let mut last_error = None;

        while Instant::now() < deadline {
            match self.http.get(&health_url).send() {
                Ok(response) if response.status().is_success() => {
                    let health: HealthResponse = response
                        .json()
                        .context("failed to parse translator health response")?;
                    if health.ok {
                        return Ok(());
                    }
                    last_error = Some(anyhow!("translator health check returned ok=false"));
                }
                Ok(response) => {
                    last_error = Some(anyhow!(
                        "translator health check returned HTTP {}",
                        response.status()
                    ));
                }
                Err(error) => {
                    last_error = Some(error.into());
                }
            }

            thread::sleep(HEALTH_POLL_INTERVAL);
        }

        Err(last_error.unwrap_or_else(|| anyhow!("translator health check timed out")))
            .context("translator server did not become healthy")
    }

    pub fn translate_to_english(&self, text: &str) -> Result<TranslateResponse> {
        let request = TranslateRequest {
            text,
            source: "auto",
            target: "en",
        };

        self.http
            .post(self.url("/translate"))
            .json(&request)
            .send()
            .context("failed to send translation request")?
            .error_for_status()
            .context("translator returned an error status")?
            .json()
            .context("failed to parse translation response")
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
    pub confidence: f64,
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
        .spawn()
        .with_context(|| {
            format!(
                "failed to launch translator sidecar with {} from {}",
                python.display(),
                translator_dir.display()
            )
        })
}

fn translator_dir() -> Result<PathBuf> {
    let local = PathBuf::from("translator");
    if local.is_dir() {
        return Ok(local);
    }

    let manifest_relative = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("translator");
    if manifest_relative.is_dir() {
        return Ok(manifest_relative);
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
    fn deserializes_translate_response_contract() {
        let response: TranslateResponse = serde_json::from_value(serde_json::json!({
            "source": "es",
            "target": "en",
            "translated_text": "[stub translation es->en] hola"
        }))
        .unwrap();

        assert_eq!(response.source, "es");
        assert_eq!(response.target, "en");
        assert_eq!(response.translated_text, "[stub translation es->en] hola");
    }
}
