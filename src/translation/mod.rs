pub mod ct2;
pub mod glossary;
pub mod model_store;
pub mod translator;

use anyhow::{Result, anyhow};
use std::{env, path::PathBuf, str::FromStr};

pub use ct2::{Ct2Config, Ct2Translator};
pub use glossary::Glossary;
pub use model_store::{ModelManifest, ModelStore, TranslationModel};
pub use translator::{
    CacheKey, CachedTranslator, NoopTranslator, TranslateRequest, TranslateResponse, Translator,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Ct2,
    Argos,
    Noop,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ct2 => "ct2",
            Self::Argos => "argos",
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
            "noop" | "none" => Ok(Self::Noop),
            other => Err(anyhow!(
                "unsupported translation backend {other:?}; expected auto, ct2, argos, or noop"
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
    pub model_dir: Option<PathBuf>,
    pub manifest_path: Option<PathBuf>,
    pub cache_capacity: usize,
    pub argos_fallback: bool,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            device: TranslationDevice::Auto,
            allow_device_fallback: false,
            model_dir: None,
            manifest_path: None,
            cache_capacity: 256,
            argos_fallback: true,
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

        if let Some(value) = read_env("TRANSLATION_ALLOW_DEVICE_FALLBACK")? {
            config.allow_device_fallback = parse_bool(&value)?;
        }

        if let Some(value) = read_env("ALBION_TRANSLATOR_ARGOS_FALLBACK")? {
            config.argos_fallback = parse_bool(&value)?;
        }

        Ok(config)
    }
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

pub fn simple_detect_language(text: &str) -> String {
    let normalized = text.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return "unknown".to_string();
    }

    let spanish_markers = [
        " el ",
        " la ",
        " los ",
        " las ",
        " que ",
        " para ",
        " por ",
        " gracias ",
        " hola ",
        " necesito ",
        " vamos ",
        " esta ",
        " donde ",
        " porque ",
        " si ",
        " no ",
    ];
    let padded = format!(" {normalized} ");

    if text.contains('¿')
        || text.contains('¡')
        || normalized.contains('ñ')
        || spanish_markers.iter().any(|marker| padded.contains(marker))
    {
        "es".to_string()
    } else if normalized.is_ascii() {
        "en".to_string()
    } else {
        "unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_backend_names() {
        assert_eq!("ct2".parse::<Backend>().unwrap(), Backend::Ct2);
        assert_eq!("ctranslate2".parse::<Backend>().unwrap(), Backend::Ct2);
        assert!("wat".parse::<Backend>().is_err());
    }

    #[test]
    fn detects_basic_spanish_for_native_path() {
        assert_eq!(simple_detect_language("hola, vamos para BZ"), "es");
    }
}
