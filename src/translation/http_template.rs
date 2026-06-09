use crate::translation::translator::{TranslateRequest, TranslateResponse, Translator};
use anyhow::{Context, Result, anyhow};
use reqwest::{Method, blocking::Client};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

const DEFAULT_METHOD: &str = "POST";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const TRANSLATEGEMMA_VLLM_PRESET: &str = include_str!("../../presets/translategemma-vllm.json");

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HttpTemplateConfig {
    pub name: String,
    pub endpoint: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    pub body: Value,
    pub response: HttpResponseConfig,
}

// TODO: Also iterate the whole message to get the most likely outcome from lingua.

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HttpResponseConfig {
    pub text_path: String,
    #[serde(default)]
    pub trim: bool,
    #[serde(default)]
    pub strip_prefixes: Vec<String>,
    #[serde(default)]
    pub truncate_before: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct HttpTemplateBackend {
    http: Client,
    config: HttpTemplateConfig,
    api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TemplateVars {
    text: String,
    src_lang: String,
    target_lang: String,
    api_key: String,
}

impl HttpTemplateConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).with_context(|| {
            format!("failed to read HTTP translation config {}", path.display())
        })?;
        serde_json::from_str(&content)
            .with_context(|| format!("failed to parse HTTP translation config {}", path.display()))
    }

    pub fn translategemma_vllm_preset() -> Result<Self> {
        serde_json::from_str(TRANSLATEGEMMA_VLLM_PRESET)
            .context("failed to parse built-in translategemma-vllm HTTP preset")
    }
}

impl HttpTemplateBackend {
    pub fn new(config: HttpTemplateConfig, api_key: Option<String>) -> Result<Self> {
        Ok(Self {
            http: Client::builder()
                .timeout(DEFAULT_TIMEOUT)
                .build()
                .context("failed to build HTTP translation client")?,
            config,
            api_key,
        })
    }

    pub fn from_path(path: impl AsRef<Path>, api_key: Option<String>) -> Result<Self> {
        Self::new(HttpTemplateConfig::from_path(path)?, api_key)
    }

    fn render_body(&self, vars: &TemplateVars) -> Value {
        interpolate_json(&self.config.body, vars)
    }
}

impl Translator for HttpTemplateBackend {
    fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse> {
        let vars = TemplateVars {
            text: request.text.clone(),
            src_lang: request.source_lang.clone(),
            target_lang: request.target_lang.clone(),
            api_key: self.api_key.clone().unwrap_or_default(),
        };
        let body = self.render_body(&vars);
        let method =
            self.config.method.parse::<Method>().with_context(|| {
                format!("invalid HTTP translation method {:?}", self.config.method)
            })?;

        let mut builder = self.http.request(method, &self.config.endpoint).json(&body);
        for (name, value) in &self.config.headers {
            if value.contains("{api_key}") && vars.api_key.is_empty() {
                continue;
            }
            builder = builder.header(name, interpolate_string(value, &vars));
        }

        let response = builder.send().with_context(|| {
            format!(
                "failed to send HTTP translation request to {}",
                self.config.endpoint
            )
        })?;
        let status = response.status();
        if !status.is_success() {
            let detail = response.text().unwrap_or_default();
            return Err(anyhow!(
                "HTTP translation endpoint returned {status}: {}",
                detail.trim()
            ));
        }

        let json: Value = response
            .json()
            .context("failed to parse HTTP translation response as JSON")?;
        let raw = extract_string_path(&json, &self.config.response.text_path)?;
        let translated_text = cleanup_text(raw, &self.config.response);

        Ok(TranslateResponse {
            translated_text,
            engine: self.config.name.clone(),
            model_id: model_id_from_body(&body),
            device: "remote".to_string(),
            detected_language: Some(request.source_lang),
            from_cache: false,
        })
    }
}

fn default_method() -> String {
    DEFAULT_METHOD.to_string()
}

fn interpolate_json(value: &Value, vars: &TemplateVars) -> Value {
    match value {
        Value::String(text) => Value::String(interpolate_string(text, vars)),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| interpolate_json(value, vars))
                .collect(),
        ),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), interpolate_json(value, vars)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn interpolate_string(template: &str, vars: &TemplateVars) -> String {
    template
        .replace("{text}", &vars.text)
        .replace("{src_lang}", &vars.src_lang)
        .replace("{target_lang}", &vars.target_lang)
        .replace("{api_key}", &vars.api_key)
}

fn extract_string_path<'a>(value: &'a Value, path: &str) -> Result<&'a str> {
    let extracted = extract_path(value, path)?;
    extracted
        .as_str()
        .ok_or_else(|| anyhow!("HTTP translation response path {path:?} is not a string"))
}

fn extract_path<'a>(mut value: &'a Value, path: &str) -> Result<&'a Value> {
    let tokens = parse_path(path)?;
    for token in tokens {
        match token {
            PathToken::Key(key) => {
                value = value.get(&key).ok_or_else(|| {
                    anyhow!("HTTP translation response path {path:?} is missing key {key:?}")
                })?;
            }
            PathToken::Index(index) => {
                value = value.get(index).ok_or_else(|| {
                    anyhow!("HTTP translation response path {path:?} is missing index {index}")
                })?;
            }
        }
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PathToken {
    Key(String),
    Index(usize),
}

fn parse_path(path: &str) -> Result<Vec<PathToken>> {
    let mut chars = path.chars().peekable();
    if chars.next() != Some('$') {
        return Err(anyhow!(
            "HTTP translation response path must start with '$'"
        ));
    }

    let mut tokens = Vec::new();
    while let Some(character) = chars.next() {
        match character {
            '.' => {
                let mut key = String::new();
                while let Some(next) = chars.peek().copied() {
                    if next == '.' || next == '[' {
                        break;
                    }
                    key.push(next);
                    chars.next();
                }
                if key.is_empty() {
                    return Err(anyhow!(
                        "HTTP translation response path contains an empty key"
                    ));
                }
                tokens.push(PathToken::Key(key));
            }
            '[' => {
                let mut digits = String::new();
                for next in chars.by_ref() {
                    if next == ']' {
                        break;
                    }
                    digits.push(next);
                }
                let index = digits.parse::<usize>().with_context(|| {
                    format!("invalid array index {digits:?} in HTTP translation response path")
                })?;
                tokens.push(PathToken::Index(index));
            }
            other => {
                return Err(anyhow!(
                    "unsupported character {other:?} in HTTP translation response path"
                ));
            }
        }
    }

    Ok(tokens)
}

fn cleanup_text(raw: &str, config: &HttpResponseConfig) -> String {
    let mut text = raw.to_string();
    for marker in &config.truncate_before {
        if marker.is_empty() {
            continue;
        }
        if let Some(index) = text.find(marker) {
            text.truncate(index);
        }
    }

    if config.trim {
        text = text.trim().to_string();
    }

    loop {
        let Some(prefix) = config
            .strip_prefixes
            .iter()
            .find(|prefix| !prefix.is_empty() && text.starts_with(prefix.as_str()))
        else {
            break;
        };
        text = text[prefix.len()..].to_string();
        if config.trim {
            text = text.trim_start().to_string();
        }
    }

    if config.trim {
        text = text.trim().to_string();
    }
    text
}

fn model_id_from_body(body: &Value) -> Option<String> {
    body.get("model")
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vars(text: &str) -> TemplateVars {
        TemplateVars {
            text: text.to_string(),
            src_lang: "pt".to_string(),
            target_lang: "en".to_string(),
            api_key: "secret".to_string(),
        }
    }

    #[test]
    fn interpolates_placeholders_recursively() {
        let value = json!({
            "prompt": "{src_lang}->{target_lang}: {text}",
            "headers": ["Bearer {api_key}"],
            "temperature": 0
        });

        assert_eq!(
            interpolate_json(&value, &vars("olá, Ava")),
            json!({
                "prompt": "pt->en: olá, Ava",
                "headers": ["Bearer secret"],
                "temperature": 0
            })
        );
    }

    #[test]
    fn translategemma_preset_generates_expected_prompt() {
        let config = HttpTemplateConfig::translategemma_vllm_preset().unwrap();
        let backend = HttpTemplateBackend::new(config, None).unwrap();
        let body = backend.render_body(&vars(
            "preciso consertar meu equipamento antes de entrar na masmorra",
        ));

        assert_eq!(
            body["prompt"],
            "<<<source>>>pt<<<target>>>en<<<text>>>preciso consertar meu equipamento antes de entrar na masmorra<<</text>>>"
        );
        assert_eq!(body["model"], "Infomaniak-AI/vllm-translategemma-4b-it");
    }

    #[test]
    fn extracts_choices_text_path() {
        let value = json!({
            "choices": [
                {
                    "text": "I need to fix my equipment before entering the dungeon"
                }
            ]
        });

        assert_eq!(
            extract_string_path(&value, "$.choices[0].text").unwrap(),
            "I need to fix my equipment before entering the dungeon"
        );
    }

    #[test]
    fn extracts_other_supported_response_shapes() {
        assert_eq!(
            extract_string_path(
                &json!({"choices": [{"message": {"content": "hello"}}]}),
                "$.choices[0].message.content"
            )
            .unwrap(),
            "hello"
        );
        assert_eq!(
            extract_string_path(&json!({"response": "hello"}), "$.response").unwrap(),
            "hello"
        );
        assert_eq!(
            extract_string_path(&json!({"translated_text": "hello"}), "$.translated_text").unwrap(),
            "hello"
        );
    }

    #[test]
    fn cleanup_truncates_markers_and_strips_leading_prefixes() {
        let config = HttpResponseConfig {
            text_path: "$.choices[0].text".to_string(),
            trim: true,
            strip_prefixes: vec![".".to_string()],
            truncate_before: vec![
                "<<<source>>>".to_string(),
                "<<<target>>>".to_string(),
                "<<</text>>>".to_string(),
            ],
        };

        assert_eq!(
            cleanup_text(
                " . I need to fix my equipment. <<<source>>>pt<<<target>>>en",
                &config
            ),
            "I need to fix my equipment."
        );
    }

    #[test]
    fn missing_response_path_returns_clear_error() {
        let error = extract_string_path(&json!({"choices": []}), "$.choices[0].text")
            .unwrap_err()
            .to_string();

        assert!(error.contains("missing index 0"), "{error}");
    }

    #[test]
    fn interpolation_preserves_punctuation_apostrophes_and_non_ascii() {
        let source = "não quebre John's \"gear\"\n{please}.";
        let value = json!({"prompt": "<<<text>>>{text}<<</text>>>"});

        let rendered = interpolate_json(&value, &vars(source));
        assert_eq!(
            rendered,
            json!({"prompt": "<<<text>>>não quebre John's \"gear\"\n{please}.<<</text>>>"})
        );

        let serialized = serde_json::to_string(&rendered).unwrap();
        assert!(serialized.contains("\\\"gear\\\""));
        assert!(serialized.contains("\\n"));
    }
}
