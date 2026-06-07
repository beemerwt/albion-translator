#[cfg(any(feature = "translation-ct2-cpu", feature = "translation-ct2-cuda"))]
use crate::translation::{Glossary, model_store::ResolvedModel};
use crate::translation::{
    ModelStore, TranslationDevice,
    model_store::is_valid_ct2_model_dir,
    translator::{TranslateRequest, TranslateResponse, Translator},
};
#[cfg(any(feature = "translation-ct2-cpu", feature = "translation-ct2-cuda"))]
use anyhow::{Context, bail};
use anyhow::{Result, anyhow};
#[cfg(any(feature = "translation-ct2-cpu", feature = "translation-ct2-cuda"))]
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Ct2Config {
    pub model_dir: Option<PathBuf>,
    pub manifest_path: Option<PathBuf>,
    pub device: TranslationDevice,
    pub allow_device_fallback: bool,
}

#[cfg(any(feature = "translation-ct2-cpu", feature = "translation-ct2-cuda"))]
mod native {
    use super::*;
    use ct2rs::{
        ComputeType, Config, Device, TranslationOptions, Translator as NativeCt2Translator,
        tokenizers::auto::Tokenizer as AutoTokenizer,
    };

    pub struct Ct2Translator {
        translator: NativeCt2Translator<AutoTokenizer>,
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
            validate_tokenizer_files(&resolved)?;

            let (translator, device) = load_translator_with_device(&resolved.path, config.device)
                .or_else(|error| {
                if config.device == TranslationDevice::Cuda && config.allow_device_fallback {
                    load_translator_with_device(&resolved.path, TranslationDevice::Cpu)
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
            })
        }
    }

    fn load_translator_with_device(
        model_path: &Path,
        requested: TranslationDevice,
    ) -> Result<(NativeCt2Translator<AutoTokenizer>, RuntimeDevice)> {
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

        NativeCt2Translator::new(model_path, &native_config)
            .with_context(|| {
                format!(
                    "failed to load CTranslate2 model from {} on {}",
                    model_path.display(),
                    device.as_str()
                )
            })
            .map(|translator| (translator, device))
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

    fn validate_tokenizer_files(resolved: &ResolvedModel) -> Result<()> {
        for relative in [
            resolved.model.tokenizer.source.as_ref(),
            resolved.model.tokenizer.target.as_ref(),
            resolved.model.tokenizer.tokenizer_json.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let path = resolved.path.join(relative);
            if !path.is_file() {
                bail!(
                    "model {} is missing tokenizer file {}",
                    resolved.model.id,
                    path.display()
                );
            }
        }

        Ok(())
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

    #[test]
    fn reports_missing_model_before_inference() {
        let config = Ct2Config {
            model_dir: Some(PathBuf::from(
                "/definitely/missing/albion-translator-models",
            )),
            manifest_path: None,
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
