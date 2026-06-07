use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const DEFAULT_MANIFEST_RELATIVE: &str = "models/manifest.json";

#[derive(Debug, Clone)]
pub struct ModelStore {
    manifest: ModelManifest,
    search_roots: Vec<PathBuf>,
}

impl ModelStore {
    pub fn load(
        manifest_path: Option<PathBuf>,
        explicit_model_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let manifest_path = manifest_path.unwrap_or_else(default_manifest_path);
        let manifest = ModelManifest::from_path(&manifest_path).with_context(|| {
            format!("failed to load model manifest {}", manifest_path.display())
        })?;
        let search_roots = model_search_roots(explicit_model_dir);

        Ok(Self {
            manifest,
            search_roots,
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn find_model(&self, source: &str, target: &str) -> Result<ResolvedModel> {
        let model = self
            .manifest
            .model_for_pair(source, target)
            .ok_or_else(|| anyhow!("unsupported translation model pair {source}->{target}"))?;

        for root in &self.search_roots {
            let path = root.join(&model.path);
            if is_valid_ct2_model_dir(&path) {
                return Ok(ResolvedModel {
                    model: model.clone(),
                    path,
                });
            }
        }

        let searched = self
            .search_roots
            .iter()
            .map(|root| root.join(&model.path).display().to_string())
            .collect::<Vec<_>>()
            .join(", ");

        Err(anyhow!(
            "CTranslate2 model {} for {source}->{target} is not installed; searched: {searched}",
            model.id
        ))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelManifest {
    pub version: u32,
    pub models: Vec<TranslationModel>,
}

impl ModelManifest {
    pub fn from_path(path: &Path) -> Result<Self> {
        let data = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str(&data).context("failed to parse translation model manifest")
    }

    pub fn model_for_pair(&self, source: &str, target: &str) -> Option<&TranslationModel> {
        self.models
            .iter()
            .find(|model| model.source == source && model.target == target)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct TranslationModel {
    pub id: String,
    pub version: String,
    pub source: String,
    pub target: String,
    pub path: PathBuf,
    pub model_type: String,
    pub tokenizer: TokenizerFiles,
    pub archive: Option<ModelArchive>,
}

impl TranslationModel {
    pub fn model_cache_key(&self) -> String {
        format!("{}@{}", self.id, self.version)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct TokenizerFiles {
    pub source: Option<PathBuf>,
    pub target: Option<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelArchive {
    pub url: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub model: TranslationModel,
    pub path: PathBuf,
}

pub fn default_manifest_path() -> PathBuf {
    if let Some(path) = bundled_models_dir().map(|dir| dir.join("manifest.json")) {
        if path.is_file() {
            return path;
        }
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_MANIFEST_RELATIVE)
}

pub fn model_search_roots(explicit_model_dir: Option<PathBuf>) -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(path) = explicit_model_dir {
        roots.push(path);
    }

    if let Some(path) = bundled_models_dir() {
        roots.push(path);
    }

    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models-cache"));

    if let Some(cache_home) = env::var_os("XDG_CACHE_HOME") {
        roots.push(
            PathBuf::from(cache_home)
                .join("albion-translator")
                .join("models"),
        );
    }

    if let Some(home) = env::var_os("HOME") {
        roots.push(
            PathBuf::from(&home)
                .join(".cache")
                .join("albion-translator")
                .join("models"),
        );
        roots.push(
            PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("albion-translator")
                .join("models"),
        );
    }

    roots
}

fn bundled_models_dir() -> Option<PathBuf> {
    env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("models")))
}

pub fn is_valid_ct2_model_dir(path: &Path) -> bool {
    path.is_dir() && path.join("model.bin").is_file() && path.join("config.json").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("albion-translator-{name}-{nonce}"))
    }

    #[test]
    fn selects_correct_model_path_from_manifest() {
        let root = test_dir("model-select");
        let manifest_path = root.join("manifest.json");
        let model_dir = root.join("cache").join("es-en");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.bin"), "").unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(
            &manifest_path,
            r#"{
              "version": 1,
              "models": [{
                "id": "opus-mt-es-en-ct2",
                "version": "2026-06-07",
                "source": "es",
                "target": "en",
                "path": "es-en",
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

        let store = ModelStore::load(Some(manifest_path), Some(root.join("cache"))).unwrap();
        let resolved = store.find_model("es", "en").unwrap();

        assert_eq!(resolved.path, model_dir);
        assert_eq!(
            resolved.model.model_cache_key(),
            "opus-mt-es-en-ct2@2026-06-07"
        );
    }

    #[test]
    fn returns_clean_error_when_model_missing() {
        let root = test_dir("model-missing");
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("manifest.json");
        fs::write(
            &manifest_path,
            r#"{
              "version": 1,
              "models": [{
                "id": "opus-mt-es-en-ct2",
                "version": "2026-06-07",
                "source": "es",
                "target": "en",
                "path": "es-en",
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

        let store = ModelStore::load(Some(manifest_path), Some(root.join("cache"))).unwrap();
        let error = store.find_model("es", "en").unwrap_err().to_string();

        assert!(error.contains("is not installed"));
        assert!(error.contains("opus-mt-es-en-ct2"));
    }
}
