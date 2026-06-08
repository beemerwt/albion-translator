use anyhow::Result;
use lingua::{Language, LanguageDetectorBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct LanguageDetection {
    pub language: String,
    pub confidence: Option<f64>,
    pub reliable: bool,
}

pub trait LanguageDetector: Send + Sync {
    fn detect(&self, text: &str) -> Result<LanguageDetection>;
}

pub struct LinguaLanguageDetector {
    detector: Arc<lingua::LanguageDetector>,
}

impl Default for LinguaLanguageDetector {
    fn default() -> Self {
        Self {
            detector: Arc::new(
                LanguageDetectorBuilder::from_languages(&[
                    Language::English,
                    Language::Spanish,
                    Language::Portuguese,
                ])
                .build(),
            ),
        }
    }
}

impl LanguageDetector for LinguaLanguageDetector {
    fn detect(&self, text: &str) -> Result<LanguageDetection> {
        let Some(language) = self.detector.detect_language_of(text) else {
            return Ok(LanguageDetection {
                language: "unknown".to_string(),
                confidence: None,
                reliable: false,
            });
        };

        let confidence = self.detector.compute_language_confidence(text, language);
        Ok(LanguageDetection {
            language: map_lang(language).to_string(),
            confidence: Some(confidence),
            reliable: confidence >= 0.65,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ManualLanguageDetector {
    detection: LanguageDetection,
}

impl ManualLanguageDetector {
    pub fn new(language: impl Into<String>, confidence: Option<f64>) -> Self {
        Self {
            detection: LanguageDetection {
                language: language.into(),
                confidence,
                reliable: confidence.is_none_or(|value| value >= 0.65),
            },
        }
    }
}

impl LanguageDetector for ManualLanguageDetector {
    fn detect(&self, _text: &str) -> Result<LanguageDetection> {
        Ok(self.detection.clone())
    }
}

#[derive(Debug, Clone, Default)]
pub struct NoopLanguageDetector;

impl LanguageDetector for NoopLanguageDetector {
    fn detect(&self, _text: &str) -> Result<LanguageDetection> {
        Ok(LanguageDetection {
            language: "unknown".to_string(),
            confidence: None,
            reliable: false,
        })
    }
}

fn map_lang(lang: Language) -> &'static str {
    match lang {
        Language::English => "en",
        Language::Spanish => "es",
        Language::Portuguese => "pt",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_detector_returns_configured_language() {
        let detector = ManualLanguageDetector::new("es", Some(0.9));
        let detection = detector.detect("anything").unwrap();

        assert_eq!(detection.language, "es");
        assert_eq!(detection.confidence, Some(0.9));
        assert!(detection.reliable);
    }

    #[test]
    fn lingua_detects_spanish_sentence() {
        let detector = LinguaLanguageDetector::default();
        let detection = detector
            .detect("hola amigo, vamos para la zona roja")
            .unwrap();

        assert_eq!(detection.language, "es");
        assert!(detection.confidence.is_some());
    }

    #[test]
    fn lingua_detects_english_sentence() {
        let detector = LinguaLanguageDetector::default();
        let detection = detector
            .detect("hello friend, we are going to the red zone")
            .unwrap();

        assert_eq!(detection.language, "en");
        assert!(detection.confidence.is_some());
    }
}
