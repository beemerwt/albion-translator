use crate::translation::{
    Ct2Config, Ct2Translator, LanguageDetection, LanguageDetector, LinguaLanguageDetector,
    ModelStore, TranslateRequest, TranslateResponse, Translator, normalize_lang,
};
use anyhow::{Context, Result, anyhow};
use std::{
    collections::HashMap,
    str::FromStr,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceLanguageConfig {
    Auto,
    Manual(String),
}

impl SourceLanguageConfig {
    pub fn resolve_override(source: &str) -> Result<Self> {
        let normalized = normalize_lang(source);
        match normalized.as_str() {
            "" | "auto" => Ok(Self::Auto),
            "es" | "pt" | "en" => Ok(Self::Manual(normalized)),
            "unknown" => Ok(Self::Manual("unknown".to_string())),
            other => Err(anyhow!("unsupported source language {other:?}")),
        }
    }
}

impl FromStr for SourceLanguageConfig {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let normalized = normalize_lang(value);
        match normalized.as_str() {
            "" | "auto" => Ok(Self::Auto),
            "es" | "pt" => Ok(Self::Manual(normalized)),
            other => Err(anyhow!(
                "unsupported TRANSLATION_SOURCE_LANG value {other:?}; expected auto, es, or pt"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TranslationOutcome {
    Translated {
        response: TranslateResponse,
        source_lang: String,
        target_lang: String,
    },
    Skipped {
        reason: TranslationSkipReason,
        source_lang: String,
        target_lang: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslationSkipReason {
    Empty,
    TooShort,
    Url,
    NumberOnly,
    EmojiOnlyOrNoText,
    AlreadyTargetLanguage,
    LowConfidence,
    UnsupportedLanguage,
    MissingModel,
}

pub trait TranslatorFactory: Send + Sync {
    fn model_id(&self, source: &str, target: &str) -> Result<String>;
    fn create(&self, source: &str, target: &str, model_id: &str) -> Result<Box<dyn Translator>>;
    fn has_installed_model_for_target(&self, target: &str) -> bool;
}

pub struct Ct2TranslatorFactory {
    config: Ct2Config,
    store: ModelStore,
    cache_capacity: usize,
}

impl Ct2TranslatorFactory {
    pub fn new(config: Ct2Config, cache_capacity: usize) -> Result<Self> {
        let store = ModelStore::load(config.manifest_path.clone(), config.model_dir.clone())?;
        Ok(Self {
            config,
            store,
            cache_capacity,
        })
    }
}

impl TranslatorFactory for Ct2TranslatorFactory {
    fn model_id(&self, source: &str, target: &str) -> Result<String> {
        Ok(self
            .store
            .find_model(source, target)
            .with_context(|| format!("no installed translation model for {source}->{target}"))?
            .model
            .model_cache_key())
    }

    fn create(&self, source: &str, target: &str, model_id: &str) -> Result<Box<dyn Translator>> {
        let translator = Ct2Translator::new(self.config.clone(), source, target)?;
        Ok(Box::new(crate::translation::CachedTranslator::new(
            translator,
            Some(model_id.to_string()),
            self.cache_capacity,
        )))
    }

    fn has_installed_model_for_target(&self, target: &str) -> bool {
        self.store.has_installed_model_for_target(target)
    }
}

pub struct TranslationRouter {
    detector: Arc<dyn LanguageDetector>,
    factory: Arc<dyn TranslatorFactory>,
    source_lang: SourceLanguageConfig,
    target_lang: String,
    confidence_threshold: f64,
    translators: Mutex<HashMap<String, Box<dyn Translator>>>,
}

impl TranslationRouter {
    pub fn new_ct2(
        config: Ct2Config,
        source_lang: SourceLanguageConfig,
        target_lang: impl Into<String>,
        confidence_threshold: f64,
        cache_capacity: usize,
    ) -> Result<Self> {
        let factory = Ct2TranslatorFactory::new(config, cache_capacity)?;
        Ok(Self::new(
            Arc::new(LinguaLanguageDetector::default()),
            Arc::new(factory),
            source_lang,
            target_lang,
            confidence_threshold,
        ))
    }

    pub fn new(
        detector: Arc<dyn LanguageDetector>,
        factory: Arc<dyn TranslatorFactory>,
        source_lang: SourceLanguageConfig,
        target_lang: impl Into<String>,
        confidence_threshold: f64,
    ) -> Self {
        Self {
            detector,
            factory,
            source_lang,
            target_lang: normalize_lang(&target_lang.into()),
            confidence_threshold,
            translators: Mutex::new(HashMap::new()),
        }
    }

    pub fn has_installed_model_for_target(&self) -> bool {
        self.factory
            .has_installed_model_for_target(&self.target_lang)
    }

    pub fn detect_language(&self, text: &str) -> Result<LanguageDetection> {
        if let Some(reason) = pre_detection_skip_reason(text) {
            return Ok(LanguageDetection {
                language: match reason {
                    TranslationSkipReason::Empty
                    | TranslationSkipReason::TooShort
                    | TranslationSkipReason::Url
                    | TranslationSkipReason::NumberOnly
                    | TranslationSkipReason::EmojiOnlyOrNoText => "unknown".to_string(),
                    _ => "unknown".to_string(),
                },
                confidence: None,
                reliable: false,
            });
        }

        self.detector.detect(text)
    }

    pub fn translate_auto(&self, text: &str) -> Result<TranslationOutcome> {
        self.translate_with_source(text, "auto")
    }

    pub fn translate_with_source(&self, text: &str, source: &str) -> Result<TranslationOutcome> {
        if let Some(reason) = pre_detection_skip_reason(text) {
            return Ok(self.skipped(reason, "unknown"));
        }

        let source_config = match SourceLanguageConfig::resolve_override(source)? {
            SourceLanguageConfig::Auto => self.source_lang.clone(),
            manual => manual,
        };

        let (source_lang, confidence) = match source_config {
            SourceLanguageConfig::Manual(language) => (language, None),
            SourceLanguageConfig::Auto => {
                let detection = self.detector.detect(text)?;
                let language = normalize_lang(&detection.language);
                if language == "unknown" {
                    return Ok(self.skipped(TranslationSkipReason::UnsupportedLanguage, language));
                }

                let confidence = detection.confidence.unwrap_or(0.0);
                if confidence < self.confidence_threshold {
                    return Ok(self.skipped(TranslationSkipReason::LowConfidence, language));
                }

                (language, Some(confidence))
            }
        };

        if source_lang == "unknown" || !is_supported_source(&source_lang) {
            return Ok(self.skipped(TranslationSkipReason::UnsupportedLanguage, source_lang));
        }

        if source_lang == self.target_lang {
            return Ok(self.skipped(TranslationSkipReason::AlreadyTargetLanguage, source_lang));
        }

        let model_id = match self.factory.model_id(&source_lang, &self.target_lang) {
            Ok(model_id) => model_id,
            Err(_) => return Ok(self.skipped(TranslationSkipReason::MissingModel, source_lang)),
        };

        let translator_key = format!("{source_lang}->{}:{model_id}", self.target_lang);
        let resolved_source_lang = source_lang.clone();
        let mut translators = self
            .translators
            .lock()
            .map_err(|_| anyhow!("translation router lock was poisoned"))?;

        if !translators.contains_key(&translator_key) {
            let translator = self
                .factory
                .create(&source_lang, &self.target_lang, &model_id)
                .with_context(|| {
                    format!(
                        "failed to create translator for {source_lang}->{}",
                        self.target_lang
                    )
                })?;
            translators.insert(translator_key.clone(), translator);
        }

        let response = translators
            .get(&translator_key)
            .expect("translator was inserted")
            .translate(TranslateRequest {
                source_lang,
                target_lang: self.target_lang.clone(),
                text: text.to_string(),
            })?;

        let _ = confidence;
        Ok(TranslationOutcome::Translated {
            response,
            source_lang: resolved_source_lang,
            target_lang: self.target_lang.clone(),
        })
    }

    fn skipped(
        &self,
        reason: TranslationSkipReason,
        source_lang: impl Into<String>,
    ) -> TranslationOutcome {
        TranslationOutcome::Skipped {
            reason,
            source_lang: source_lang.into(),
            target_lang: self.target_lang.clone(),
        }
    }
}

pub fn normalized_cache_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn pre_detection_skip_reason(text: &str) -> Option<TranslationSkipReason> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some(TranslationSkipReason::Empty);
    }

    if contains_url(trimmed) {
        return Some(TranslationSkipReason::Url);
    }

    let alphabetic_count = trimmed
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    if alphabetic_count == 0 {
        if trimmed.chars().any(|character| character.is_numeric()) {
            Some(TranslationSkipReason::NumberOnly)
        } else {
            Some(TranslationSkipReason::EmojiOnlyOrNoText)
        }
    } else if alphabetic_count < 4 {
        Some(TranslationSkipReason::TooShort)
    } else {
        None
    }
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

fn is_supported_source(language: &str) -> bool {
    matches!(language, "en" | "es" | "pt")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translation::{ManualLanguageDetector, TranslateResponse};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct FakeFactory {
        models: HashMap<String, String>,
        creates: AtomicUsize,
    }

    impl FakeFactory {
        fn with_model(source: &str, target: &str, model_id: &str) -> Self {
            Self {
                models: HashMap::from([(format!("{source}->{target}"), model_id.to_string())]),
                creates: AtomicUsize::new(0),
            }
        }
    }

    impl TranslatorFactory for FakeFactory {
        fn model_id(&self, source: &str, target: &str) -> Result<String> {
            self.models
                .get(&format!("{source}->{target}"))
                .cloned()
                .ok_or_else(|| anyhow!("missing fake model"))
        }

        fn create(
            &self,
            _source: &str,
            _target: &str,
            model_id: &str,
        ) -> Result<Box<dyn Translator>> {
            self.creates.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeTranslator {
                model_id: model_id.to_string(),
            }))
        }

        fn has_installed_model_for_target(&self, target: &str) -> bool {
            self.models
                .keys()
                .any(|key| key.ends_with(&format!("->{target}")))
        }
    }

    struct FakeTranslator {
        model_id: String,
    }

    impl Translator for FakeTranslator {
        fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse> {
            Ok(TranslateResponse {
                translated_text: format!("{} -> {}", request.text, request.target_lang),
                engine: "fake".to_string(),
                model_id: Some(self.model_id.clone()),
                device: "cpu".to_string(),
                detected_language: Some(request.source_lang),
                from_cache: false,
            })
        }
    }

    fn router_with(
        language: &str,
        confidence: f64,
        factory: Arc<FakeFactory>,
        source_lang: SourceLanguageConfig,
    ) -> TranslationRouter {
        TranslationRouter::new(
            Arc::new(ManualLanguageDetector::new(language, Some(confidence))),
            factory,
            source_lang,
            "en",
            0.65,
        )
    }

    #[test]
    fn skips_very_short_messages() {
        let router = router_with(
            "es",
            0.9,
            Arc::new(FakeFactory::with_model("es", "en", "es-en")),
            SourceLanguageConfig::Auto,
        );

        assert!(matches!(
            router.translate_auto("gg").unwrap(),
            TranslationOutcome::Skipped {
                reason: TranslationSkipReason::TooShort,
                ..
            }
        ));
    }

    #[test]
    fn skips_urls() {
        let router = router_with(
            "es",
            0.9,
            Arc::new(FakeFactory::with_model("es", "en", "es-en")),
            SourceLanguageConfig::Auto,
        );

        assert!(matches!(
            router.translate_auto("mira https://example.com").unwrap(),
            TranslationOutcome::Skipped {
                reason: TranslationSkipReason::Url,
                ..
            }
        ));
    }

    #[test]
    fn skips_numbers_and_emoji_only_text() {
        let router = router_with(
            "es",
            0.9,
            Arc::new(FakeFactory::with_model("es", "en", "es-en")),
            SourceLanguageConfig::Auto,
        );

        assert!(matches!(
            router.translate_auto("12345").unwrap(),
            TranslationOutcome::Skipped {
                reason: TranslationSkipReason::NumberOnly,
                ..
            }
        ));
        assert!(matches!(
            router.translate_auto("!!!").unwrap(),
            TranslationOutcome::Skipped {
                reason: TranslationSkipReason::EmojiOnlyOrNoText,
                ..
            }
        ));
    }

    #[test]
    fn skips_already_target_language() {
        let router = router_with(
            "en",
            0.9,
            Arc::new(FakeFactory::with_model("es", "en", "es-en")),
            SourceLanguageConfig::Auto,
        );

        assert!(matches!(
            router.translate_auto("hello friend").unwrap(),
            TranslationOutcome::Skipped {
                reason: TranslationSkipReason::AlreadyTargetLanguage,
                ..
            }
        ));
    }

    #[test]
    fn skips_low_confidence_auto_detection() {
        let router = router_with(
            "es",
            0.2,
            Arc::new(FakeFactory::with_model("es", "en", "es-en")),
            SourceLanguageConfig::Auto,
        );

        assert!(matches!(
            router.translate_auto("hola amigo vamos").unwrap(),
            TranslationOutcome::Skipped {
                reason: TranslationSkipReason::LowConfidence,
                ..
            }
        ));
    }

    #[test]
    fn manual_source_override_bypasses_detection() {
        let factory = Arc::new(FakeFactory::with_model("es", "en", "es-en"));
        let router = router_with(
            "en",
            0.99,
            factory,
            SourceLanguageConfig::Manual("es".to_string()),
        );
        let outcome = router.translate_auto("hola amigo vamos").unwrap();

        assert!(matches!(outcome, TranslationOutcome::Translated { .. }));
    }

    #[test]
    fn missing_model_skips_cleanly() {
        let router = router_with(
            "pt",
            0.99,
            Arc::new(FakeFactory::with_model("es", "en", "es-en")),
            SourceLanguageConfig::Manual("pt".to_string()),
        );

        assert!(matches!(
            router.translate_auto("ola amigo vamos").unwrap(),
            TranslationOutcome::Skipped {
                reason: TranslationSkipReason::MissingModel,
                ..
            }
        ));
    }

    #[test]
    fn unsupported_detected_language_does_not_create_translator() {
        let factory = Arc::new(FakeFactory::with_model("es", "en", "es-en"));
        let router = router_with("unknown", 0.99, factory.clone(), SourceLanguageConfig::Auto);

        assert!(matches!(
            router.translate_auto("bonjour mon ami").unwrap(),
            TranslationOutcome::Skipped {
                reason: TranslationSkipReason::UnsupportedLanguage,
                ..
            }
        ));
        assert_eq!(factory.creates.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn normalizes_cache_text_by_collapsing_whitespace() {
        assert_eq!(normalized_cache_text(" hola   Ava  "), "hola Ava");
    }
}
