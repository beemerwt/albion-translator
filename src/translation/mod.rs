pub mod cache;
pub mod ct2;
pub mod glossary;
pub mod http_template;
pub mod language_detector;
pub mod model_store;
pub mod router;
pub mod translator;

use anyhow::{Result, anyhow};
use std::{env, path::PathBuf, str::FromStr};

pub use cache::{CachedTranslation, TranslationCache};
pub use ct2::{Ct2Config, Ct2Translator};
pub use glossary::Glossary;
pub use http_template::{HttpTemplateBackend, HttpTemplateConfig};
pub use language_detector::{
    LanguageDetection, LanguageDetector, LinguaLanguageDetector, ManualLanguageDetector,
    NoopLanguageDetector,
};
pub use model_store::{ModelManifest, ModelStore, TranslationModel};
pub use router::{
    Ct2TranslatorFactory, SourceLanguageConfig, TranslationOutcome, TranslationRouter,
    TranslationSkipReason,
};
pub use translator::{
    CacheKey, CachedTranslator, NoopTranslator, TranslateRequest, TranslateResponse, Translator,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Ct2,
    Argos,
    Google,
    Http,
    TranslateGemmaVllm,
    Noop,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ct2 => "ct2",
            Self::Argos => "argos",
            Self::Google => "google",
            Self::Http => "http",
            Self::TranslateGemmaVllm => "translategemma-vllm",
            Self::Noop => "noop",
        }
    }
}

impl FromStr for Backend {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "ct2" | "ctranslate2" => Ok(Self::Ct2),
            "argos" => Ok(Self::Argos),
            "google" | "google-translate" | "google_translate" => Ok(Self::Google),
            "http" | "custom-http" | "model-http" | "http-template" => Ok(Self::Http),
            "translategemma-vllm" | "translate-gemma-vllm" | "translategemma_vllm" => {
                Ok(Self::TranslateGemmaVllm)
            }
            "noop" | "none" => Ok(Self::Noop),
            other => Err(anyhow!(
                "unsupported translation backend {other:?}; expected auto, ct2, argos, google, http, translategemma-vllm, or noop"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslationDevice {
    Auto,
    Cpu,
    Cuda,
}

impl TranslationDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

impl FromStr for TranslationDevice {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => Err(anyhow!(
                "unsupported translation device {other:?}; expected auto, cpu, or cuda"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranslationConfig {
    pub backend: Backend,
    pub device: TranslationDevice,
    pub allow_device_fallback: bool,
    pub source_lang: SourceLanguageConfig,
    pub target_lang: String,
    pub detection_confidence_threshold: f64,
    pub model_dir: Option<PathBuf>,
    pub manifest_path: Option<PathBuf>,
    pub cache_capacity: usize,
    pub cache_db_path: PathBuf,
    pub argos_fallback: bool,
    pub http_config_path: Option<PathBuf>,
    pub http_api_key: Option<String>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            device: TranslationDevice::Auto,
            allow_device_fallback: false,
            source_lang: SourceLanguageConfig::Auto,
            target_lang: "en".to_string(),
            detection_confidence_threshold: 0.65,
            model_dir: None,
            manifest_path: None,
            cache_capacity: 256,
            cache_db_path: PathBuf::from("translations.sqlite3"),
            argos_fallback: true,
            http_config_path: None,
            http_api_key: None,
        }
    }
}

impl TranslationConfig {
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();

        if let Some(value) = read_env("TRANSLATION_BACKEND")? {
            config.backend = value.parse()?;
        } else if let Some(value) = read_env("ALBION_TRANSLATION_BACKEND")? {
            config.backend = value.parse()?;
        }

        if let Some(value) = read_env("TRANSLATION_DEVICE")? {
            config.device = value.parse()?;
        }

        if let Some(value) = read_env("TRANSLATION_SOURCE_LANG")? {
            config.source_lang = value.parse()?;
        }

        if let Some(value) = read_env("TRANSLATION_TARGET_LANG")? {
            config.target_lang = parse_target_lang(&value)?;
        }

        if let Some(value) = read_env("TRANSLATION_DETECTION_CONFIDENCE_THRESHOLD")? {
            config.detection_confidence_threshold = value.parse().map_err(|error| {
                anyhow!(
                    "invalid TRANSLATION_DETECTION_CONFIDENCE_THRESHOLD value {value:?}: {error}"
                )
            })?;
            if !(0.0..=1.0).contains(&config.detection_confidence_threshold) {
                return Err(anyhow!(
                    "TRANSLATION_DETECTION_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0"
                ));
            }
        }

        if let Some(value) = read_env("TRANSLATION_MODEL_DIR")? {
            config.model_dir = Some(PathBuf::from(value));
        } else if let Some(value) = read_env("ALBION_TRANSLATION_MODEL_DIR")? {
            config.model_dir = Some(PathBuf::from(value));
        }

        if let Some(value) = read_env("TRANSLATION_MODEL_MANIFEST")? {
            config.manifest_path = Some(PathBuf::from(value));
        } else if let Some(value) = read_env("ALBION_TRANSLATION_MODEL_MANIFEST")? {
            config.manifest_path = Some(PathBuf::from(value));
        }

        if let Some(value) = read_env("TRANSLATION_CACHE_CAPACITY")? {
            config.cache_capacity = value.parse().map_err(|error| {
                anyhow!("invalid TRANSLATION_CACHE_CAPACITY value {value:?}: {error}")
            })?;
        }

        if let Some(value) = read_env("TRANSLATION_CACHE_DB")? {
            config.cache_db_path = PathBuf::from(value);
        } else if let Some(value) = read_env("ALBION_TRANSLATION_CACHE_DB")? {
            config.cache_db_path = PathBuf::from(value);
        }

        if let Some(value) = read_env("TRANSLATION_ALLOW_DEVICE_FALLBACK")? {
            config.allow_device_fallback = parse_bool(&value)?;
        }

        if let Some(value) = read_env("ALBION_TRANSLATOR_ARGOS_FALLBACK")? {
            config.argos_fallback = parse_bool(&value)?;
        }

        if let Some(value) = read_env("TRANSLATION_HTTP_CONFIG")? {
            config.http_config_path = Some(PathBuf::from(value));
        } else if let Some(value) = read_env("ALBION_TRANSLATION_HTTP_CONFIG")? {
            config.http_config_path = Some(PathBuf::from(value));
        }

        if let Some(value) = read_env("TRANSLATION_HTTP_API_KEY")? {
            config.http_api_key = Some(value);
        } else if let Some(value) = read_env("ALBION_TRANSLATION_HTTP_API_KEY")? {
            config.http_api_key = Some(value);
        }

        Ok(config)
    }
}

fn parse_target_lang(value: &str) -> Result<String> {
    let normalized = normalize_lang(value);
    if normalized.is_empty() {
        Err(anyhow!("TRANSLATION_TARGET_LANG cannot be empty"))
    } else {
        Ok(normalized)
    }
}

pub(crate) fn normalize_lang(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

fn read_env(name: &str) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).map_err(|error| anyhow!("failed to read {name}: {error}")),
    }
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(anyhow!("invalid boolean value {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_backend_names() {
        assert_eq!("ct2".parse::<Backend>().unwrap(), Backend::Ct2);
        assert_eq!("ctranslate2".parse::<Backend>().unwrap(), Backend::Ct2);
        assert_eq!("google".parse::<Backend>().unwrap(), Backend::Google);
        assert_eq!("http".parse::<Backend>().unwrap(), Backend::Http);
        assert_eq!(
            "translategemma-vllm".parse::<Backend>().unwrap(),
            Backend::TranslateGemmaVllm
        );
        assert!("wat".parse::<Backend>().is_err());
    }

    #[test]
    fn parses_source_language_config() {
        assert_eq!(
            "auto".parse::<SourceLanguageConfig>().unwrap(),
            SourceLanguageConfig::Auto
        );
        assert_eq!(
            "es".parse::<SourceLanguageConfig>().unwrap(),
            SourceLanguageConfig::Manual("es".to_string())
        );
        assert!("fr".parse::<SourceLanguageConfig>().is_err());
    }
}
