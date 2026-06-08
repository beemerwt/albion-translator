#[cfg(any(feature = "translation-ct2-cpu", feature = "translation-ct2-cuda"))]
use crate::translation::Glossary;
use crate::translation::{
    ModelStore, TranslationDevice,
    model_store::{ResolvedModel, is_valid_ct2_model_dir},
    translator::{TranslateRequest, TranslateResponse, Translator},
};
#[cfg(any(feature = "translation-ct2-cpu", feature = "translation-ct2-cuda"))]
use anyhow::Context;
use anyhow::{Result, anyhow, bail};
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Ct2Config {
    pub model_dir: Option<PathBuf>,
    pub manifest_path: Option<PathBuf>,
    pub device: TranslationDevice,
    pub allow_device_fallback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenizerConfig {
    SentencePiece { source: PathBuf, target: PathBuf },
    HuggingFace { tokenizer_json: PathBuf },
    Auto,
}

fn tokenizer_config_for_model(resolved: &ResolvedModel) -> Result<TokenizerConfig> {
    match (
        resolved.model.tokenizer.source.as_ref(),
        resolved.model.tokenizer.target.as_ref(),
    ) {
        (Some(source), Some(target)) => {
            let source = resolved.path.join(source);
            let target = resolved.path.join(target);
            ensure_tokenizer_file(&resolved.model.id, &source)?;
            ensure_tokenizer_file(&resolved.model.id, &target)?;
            return Ok(TokenizerConfig::SentencePiece { source, target });
        }
        (Some(_), None) | (None, Some(_)) => {
            bail!(
                "model {} declares only one SentencePiece tokenizer file; both source and target are required",
                resolved.model.id
            );
        }
        (None, None) => {}
    }

    if let Some(tokenizer_json) = resolved.model.tokenizer.tokenizer_json.as_ref() {
        let tokenizer_json = resolved.path.join(tokenizer_json);
        ensure_tokenizer_file(&resolved.model.id, &tokenizer_json)?;
        return Ok(TokenizerConfig::HuggingFace { tokenizer_json });
    }

    Ok(TokenizerConfig::Auto)
}

fn ensure_tokenizer_file(model_id: &str, path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        bail!(
            "model {model_id} is missing tokenizer file {}",
            path.display()
        )
    }
}

#[cfg(any(feature = "translation-ct2-cpu", feature = "translation-ct2-cuda"))]
mod native {
    use super::*;
    use ct2rs::{
        ComputeType, Config, Device, TranslationOptions, Translator as NativeCt2Translator,
        tokenizers::{
            auto::Tokenizer as AutoTokenizer, hf::Tokenizer as HfTokenizer,
            sentencepiece::Tokenizer as SentencePieceTokenizer,
        },
    };

    pub struct Ct2Translator {
        translator: Ct2TranslatorBackend,
        model_id: String,
        source_lang: String,
        target_lang: String,
        device: RuntimeDevice,
        glossary: Glossary,
    }

    impl Ct2Translator {
        pub fn new(config: Ct2Config, source: &str, target: &str) -> Result<Self> {
            let store = ModelStore::load(config.manifest_path, config.model_dir)?;
            let resolved = store.find_model(source, target)?;

            let tokenizer_config = tokenizer_config_for_model(&resolved).with_context(|| {
                format!("invalid tokenizer config for model {}", resolved.model.id)
            })?;

            let (translator, device) =
                load_translator_with_device(&resolved.path, &tokenizer_config, config.device)
                    .or_else(|error| {
                        if config.device == TranslationDevice::Cuda && config.allow_device_fallback
                        {
                            load_translator_with_device(
                                &resolved.path,
                                &tokenizer_config,
                                TranslationDevice::Cpu,
                            )
                            .with_context(|| format!("CUDA load failed first: {error}"))
                        } else {
                            Err(error)
                        }
                    })?;

            Ok(Self {
                translator,
                model_id: resolved.model.model_cache_key(),
                source_lang: resolved.model.source,
                target_lang: resolved.model.target,
                device,
                glossary: Glossary::default(),
            })
        }

        pub fn is_available(config: &Ct2Config, source: &str, target: &str) -> bool {
            ModelStore::load(config.manifest_path.clone(), config.model_dir.clone())
                .and_then(|store| store.find_model(source, target))
                .map(|resolved| is_valid_ct2_model_dir(&resolved.path))
                .unwrap_or(false)
        }

        pub fn model_id(&self) -> Option<String> {
            Some(self.model_id.clone())
        }

        pub fn device(&self) -> &'static str {
            self.device.as_str()
        }

        pub fn cuda_build_available() -> bool {
            cfg!(feature = "translation-ct2-cuda")
        }
    }

    impl Translator for Ct2Translator {
        fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse> {
            if request.source_lang != self.source_lang || request.target_lang != self.target_lang {
                bail!(
                    "ct2 model {} supports {}->{}, not {}->{}",
                    self.model_id,
                    self.source_lang,
                    self.target_lang,
                    request.source_lang,
                    request.target_lang
                );
            }

            let protected = self.glossary.protect(&request.text);
            let mut options: TranslationOptions<String, String> = TranslationOptions {
                beam_size: 1,
                max_batch_size: 1,
                max_input_length: 128,
                max_decoding_length: 128,
                ..Default::default()
            };
            options.return_scores = false;

            let results = self
                .translator
                .translate_batch(&[protected.text.as_str()], &options, None)
                .context("ct2rs translation failed")?;
            let translated = results
                .into_iter()
                .next()
                .map(|(text, _score)| text)
                .ok_or_else(|| anyhow!("ct2rs returned no translation result"))?;

            Ok(TranslateResponse {
                translated_text: protected.restore(translated.trim()),
                engine: "ct2".to_string(),
                model_id: Some(self.model_id.clone()),
                device: self.device.as_str().to_string(),
                detected_language: Some(request.source_lang),
                from_cache: false,
            })
        }
    }

    fn load_translator_with_device(
        model_path: &Path,
        tokenizer_config: &TokenizerConfig,
        requested: TranslationDevice,
    ) -> Result<(Ct2TranslatorBackend, RuntimeDevice)> {
        let device = resolve_runtime_device(requested)?;
        let native_config = Config {
            device: match device {
                RuntimeDevice::Cpu => Device::CPU,
                RuntimeDevice::Cuda => Device::CUDA,
            },
            compute_type: ComputeType::AUTO,
            device_indices: vec![0],
            tensor_parallel: false,
            num_threads_per_replica: 0,
            max_queued_batches: 1,
            cpu_core_offset: -1,
        };

        let translator = Ct2TranslatorBackend::load(model_path, tokenizer_config, &native_config)
            .with_context(|| {
            format!(
                "failed to load CTranslate2 model from {} on {} with {} tokenizer",
                model_path.display(),
                device.as_str(),
                tokenizer_config.name()
            )
        })?;

        Ok((translator, device))
    }

    fn resolve_runtime_device(requested: TranslationDevice) -> Result<RuntimeDevice> {
        match requested {
            TranslationDevice::Cpu => Ok(RuntimeDevice::Cpu),
            TranslationDevice::Cuda if cfg!(feature = "translation-ct2-cuda") => {
                Ok(RuntimeDevice::Cuda)
            }
            TranslationDevice::Cuda => Err(anyhow!(
                "TRANSLATION_DEVICE=cuda requires a binary built with --features translation-ct2-cuda"
            )),
            TranslationDevice::Auto if cfg!(feature = "translation-ct2-cuda") => {
                Ok(RuntimeDevice::Cuda)
            }
            TranslationDevice::Auto => Ok(RuntimeDevice::Cpu),
        }
    }

    enum Ct2TranslatorBackend {
        Auto(NativeCt2Translator<AutoTokenizer>),
        SentencePiece(NativeCt2Translator<SentencePieceTokenizer>),
        HuggingFace(NativeCt2Translator<HfTokenizer>),
    }

    impl Ct2TranslatorBackend {
        fn load(
            model_path: &Path,
            tokenizer_config: &TokenizerConfig,
            native_config: &Config,
        ) -> Result<Self> {
            match tokenizer_config {
                TokenizerConfig::SentencePiece { source, target } => {
                    let tokenizer = SentencePieceTokenizer::from_file(source, target)
                        .with_context(|| {
                            format!(
                                "failed to load SentencePiece tokenizer files {} and {}",
                                source.display(),
                                target.display()
                            )
                        })?;
                    NativeCt2Translator::with_tokenizer(model_path, tokenizer, native_config)
                        .map(Self::SentencePiece)
                }
                TokenizerConfig::HuggingFace { tokenizer_json } => {
                    let tokenizer = HfTokenizer::from_file(tokenizer_json).with_context(|| {
                        format!(
                            "failed to load Hugging Face tokenizer file {}",
                            tokenizer_json.display()
                        )
                    })?;
                    NativeCt2Translator::with_tokenizer(model_path, tokenizer, native_config)
                        .map(Self::HuggingFace)
                }
                TokenizerConfig::Auto => {
                    NativeCt2Translator::new(model_path, native_config).map(Self::Auto)
                }
            }
        }

        fn translate_batch<U, V, W>(
            &self,
            sources: &[U],
            options: &TranslationOptions<V, W>,
            callback: Option<&mut dyn FnMut(ct2rs::GenerationStepResult) -> Result<()>>,
        ) -> Result<Vec<(String, Option<f32>)>>
        where
            U: AsRef<str>,
            V: AsRef<str>,
            W: AsRef<str>,
        {
            match self {
                Self::Auto(translator) => translator.translate_batch(sources, options, callback),
                Self::SentencePiece(translator) => {
                    translator.translate_batch(sources, options, callback)
                }
                Self::HuggingFace(translator) => {
                    translator.translate_batch(sources, options, callback)
                }
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum RuntimeDevice {
        Cpu,
        Cuda,
    }

    impl RuntimeDevice {
        fn as_str(self) -> &'static str {
            match self {
                Self::Cpu => "cpu",
                Self::Cuda => "cuda",
            }
        }
    }
}

impl TokenizerConfig {
    fn name(&self) -> &'static str {
        match self {
            Self::SentencePiece { .. } => "sentencepiece",
            Self::HuggingFace { .. } => "huggingface",
            Self::Auto => "auto",
        }
    }
}

#[cfg(not(any(feature = "translation-ct2-cpu", feature = "translation-ct2-cuda")))]
mod native {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Ct2Translator {
        model_id: Option<String>,
    }

    impl Ct2Translator {
        pub fn new(_config: Ct2Config, _source: &str, _target: &str) -> Result<Self> {
            Err(anyhow!(
                "ct2 translation support is not compiled in; rebuild with --features translation-ct2-cpu or translation-ct2-cuda"
            ))
        }

        pub fn is_available(config: &Ct2Config, source: &str, target: &str) -> bool {
            ModelStore::load(config.manifest_path.clone(), config.model_dir.clone())
                .and_then(|store| store.find_model(source, target))
                .map(|resolved| is_valid_ct2_model_dir(&resolved.path))
                .unwrap_or(false)
        }

        pub fn model_id(&self) -> Option<String> {
            self.model_id.clone()
        }

        pub fn device(&self) -> &'static str {
            "unavailable"
        }

        pub fn cuda_build_available() -> bool {
            false
        }
    }

    impl Translator for Ct2Translator {
        fn translate(&self, _request: TranslateRequest) -> Result<TranslateResponse> {
            Err(anyhow!(
                "ct2 translation support is not compiled in; rebuild with --features translation-ct2-cpu or translation-ct2-cuda"
            ))
        }
    }
}

pub use native::Ct2Translator;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translation::model_store::{TokenizerFiles, TranslationModel};

    #[test]
    fn reports_missing_model_before_inference() {
        let root = std::env::temp_dir().join(format!(
            "albion-translator-missing-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("manifest.json");
        std::fs::write(
            &manifest_path,
            r#"{
              "version": 1,
              "models": [{
                "id": "missing-es-en",
                "version": "test",
                "source": "es",
                "target": "en",
                "path": "missing-es-en",
                "model_type": "marian",
                "tokenizer": {
                  "source": "source.spm",
                  "target": "target.spm",
                  "tokenizer_json": null
                },
                "archive": null
              }]
            }"#,
        )
        .unwrap();

        let config = Ct2Config {
            model_dir: Some(root.join("cache")),
            manifest_path: Some(manifest_path),
            device: TranslationDevice::Cpu,
            allow_device_fallback: false,
        };

        let error = match Ct2Translator::new(config, "es", "en") {
            Ok(translator) => translator
                .translate(TranslateRequest {
                    source_lang: "es".to_string(),
                    target_lang: "en".to_string(),
                    text: "hola mundo".to_string(),
                })
                .unwrap_err()
                .to_string(),
            Err(err) => err.to_string(),
        };

        assert!(
            error.contains("not installed")
                || error.contains("ct2 translation support is not compiled in")
        );
    }

    #[test]
    fn uses_manifest_sentencepiece_tokenizer_filenames() {
        let root = std::env::temp_dir().join(format!(
            "albion-translator-tokenizer-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let model_path = root.join("quickmt-pt-en");
        std::fs::create_dir_all(&model_path).unwrap();
        std::fs::write(model_path.join("src.spm.model"), "").unwrap();
        std::fs::write(model_path.join("tgt.spm.model"), "").unwrap();

        let resolved = ResolvedModel {
            model: TranslationModel {
                id: "quickmt-pt-en-ct2".to_string(),
                version: "test".to_string(),
                source: "pt".to_string(),
                target: "en".to_string(),
                path: "quickmt-pt-en".into(),
                model_type: "quickmt_ct2".to_string(),
                tokenizer: TokenizerFiles {
                    source: Some("src.spm.model".into()),
                    target: Some("tgt.spm.model".into()),
                    tokenizer_json: None,
                },
                archive: None,
            },
            path: model_path.clone(),
        };

        assert_eq!(
            tokenizer_config_for_model(&resolved).unwrap(),
            TokenizerConfig::SentencePiece {
                source: model_path.join("src.spm.model"),
                target: model_path.join("tgt.spm.model"),
            }
        );
    }

    #[test]
    #[ignore = "requires a local es->en CTranslate2 model"]
    fn smoke_translate_spanish_to_english() {
        let config = Ct2Config {
            model_dir: std::env::var_os("TRANSLATION_MODEL_DIR").map(PathBuf::from),
            manifest_path: std::env::var_os("TRANSLATION_MODEL_MANIFEST").map(PathBuf::from),
            device: TranslationDevice::Cpu,
            allow_device_fallback: false,
        };
        let translator = Ct2Translator::new(config, "es", "en").unwrap();
        let response = translator
            .translate(TranslateRequest {
                source_lang: "es".to_string(),
                target_lang: "en".to_string(),
                text: "hola mundo".to_string(),
            })
            .unwrap();

        assert_eq!(response.engine, "ct2");
        assert_eq!(response.device, "cpu");
        assert!(!response.translated_text.trim().is_empty());
    }

    #[test]
    #[ignore = "requires a local pt->en CTranslate2 model"]
    fn smoke_translate_portuguese_to_english() {
        let config = Ct2Config {
            model_dir: std::env::var_os("TRANSLATION_MODEL_DIR").map(PathBuf::from),
            manifest_path: std::env::var_os("TRANSLATION_MODEL_MANIFEST").map(PathBuf::from),
            device: TranslationDevice::Cpu,
            allow_device_fallback: false,
        };
        let translator = Ct2Translator::new(config, "pt", "en").unwrap();
        let response = translator
            .translate(TranslateRequest {
                source_lang: "pt".to_string(),
                target_lang: "en".to_string(),
                text: "olá mundo".to_string(),
            })
            .unwrap();

        assert_eq!(response.engine, "ct2");
        assert_eq!(response.device, "cpu");
        assert!(!response.translated_text.trim().is_empty());
    }

    #[test]
    #[cfg(feature = "translation-ct2-cuda")]
    #[ignore = "requires a CUDA ct2rs build, CUDA runtime, GPU, and a local es->en model"]
    fn smoke_translate_spanish_to_english_cuda() {
        let config = Ct2Config {
            model_dir: std::env::var_os("TRANSLATION_MODEL_DIR").map(PathBuf::from),
            manifest_path: std::env::var_os("TRANSLATION_MODEL_MANIFEST").map(PathBuf::from),
            device: TranslationDevice::Cuda,
            allow_device_fallback: false,
        };
        let translator = Ct2Translator::new(config, "es", "en").unwrap();

        assert_eq!(translator.device(), "cuda");
    }
}
