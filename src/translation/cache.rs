use crate::{translation::router::normalized_cache_text, translator::TranslateResponse};
use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, OptionalExtension, params};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

const GOOGLE_BACKEND: &str = "google";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedTranslation {
    pub normalized_source_text: String,
    pub original_source_text: String,
    pub translated_text: String,
    pub target_language: String,
    pub detected_source_language: Option<String>,
    pub backend: String,
}

#[derive(Debug, Clone)]
pub struct TranslationCache {
    connection: Arc<Mutex<Connection>>,
}

impl TranslationCache {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = Connection::open(path.as_ref()).with_context(|| {
            format!(
                "failed to open translation cache database {}",
                path.as_ref().display()
            )
        })?;
        Self::from_connection(connection)
    }

    pub fn in_memory() -> Result<Self> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(connection: Connection) -> Result<Self> {
        let cache = Self {
            connection: Arc::new(Mutex::new(connection)),
        };
        cache.init()?;
        Ok(cache)
    }

    fn init(&self) -> Result<()> {
        self.connection()?.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS translation_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                normalized_source_text TEXT NOT NULL,
                original_source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                target_language TEXT NOT NULL,
                detected_source_language TEXT,
                backend TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(backend, normalized_source_text, target_language)
            );

            CREATE INDEX IF NOT EXISTS idx_translation_cache_lookup
                ON translation_cache (backend, normalized_source_text, target_language);
            CREATE INDEX IF NOT EXISTS idx_translation_cache_last_used
                ON translation_cache (last_used_at);
            ",
        )?;
        Ok(())
    }

    pub fn lookup_google(
        &self,
        text: &str,
        target_language: &str,
    ) -> Result<Option<TranslateResponse>> {
        let normalized = normalized_cache_text(text);
        let connection = self.connection()?;
        let cached = connection
            .query_row(
                "
                SELECT normalized_source_text,
                       original_source_text,
                       translated_text,
                       target_language,
                       detected_source_language,
                       backend
                FROM translation_cache
                WHERE backend = ?1
                  AND normalized_source_text = ?2
                  AND target_language = ?3
                ",
                params![GOOGLE_BACKEND, normalized, target_language],
                |row| {
                    Ok(CachedTranslation {
                        normalized_source_text: row.get(0)?,
                        original_source_text: row.get(1)?,
                        translated_text: row.get(2)?,
                        target_language: row.get(3)?,
                        detected_source_language: row.get(4)?,
                        backend: row.get(5)?,
                    })
                },
            )
            .optional()?;

        let Some(cached) = cached else {
            return Ok(None);
        };

        connection.execute(
            "
            UPDATE translation_cache
            SET last_used_at = CURRENT_TIMESTAMP
            WHERE backend = ?1
              AND normalized_source_text = ?2
              AND target_language = ?3
            ",
            params![GOOGLE_BACKEND, normalized, target_language],
        )?;

        Ok(Some(TranslateResponse {
            source: cached
                .detected_source_language
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            target: cached.target_language,
            translated_text: cached.translated_text,
            engine: cached.backend,
            model_id: None,
            device: "remote".to_string(),
            detected_language: cached.detected_source_language,
            from_cache: true,
        }))
    }

    pub fn insert_google(&self, original_text: &str, response: &TranslateResponse) -> Result<()> {
        if response.engine != GOOGLE_BACKEND {
            return Ok(());
        }

        let normalized = normalized_cache_text(original_text);
        self.connection()?.execute(
            "
            INSERT INTO translation_cache (
                normalized_source_text,
                original_source_text,
                translated_text,
                target_language,
                detected_source_language,
                backend
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(backend, normalized_source_text, target_language)
            DO UPDATE SET
                original_source_text = excluded.original_source_text,
                translated_text = excluded.translated_text,
                detected_source_language = excluded.detected_source_language,
                updated_at = CURRENT_TIMESTAMP,
                last_used_at = CURRENT_TIMESTAMP
            ",
            params![
                normalized,
                original_text,
                response.translated_text,
                response.target,
                response
                    .detected_language
                    .as_deref()
                    .or(Some(response.source.as_str())),
                GOOGLE_BACKEND,
            ],
        )?;
        Ok(())
    }

    fn connection(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.connection
            .lock()
            .map_err(|_| anyhow!("translation cache lock was poisoned"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn cache_hit_reuses_normalized_google_translation() {
        let cache = TranslationCache::in_memory().unwrap();
        cache
            .insert_google(
                " hola   amigo ",
                &google_response("hello friend", Some("es")),
            )
            .unwrap();

        let cached = cache.lookup_google("hola amigo", "en").unwrap().unwrap();

        assert_eq!(cached.translated_text, "hello friend");
        assert_eq!(cached.detected_language.as_deref(), Some("es"));
        assert!(cached.from_cache);
    }

    #[test]
    fn non_google_results_are_not_inserted() {
        let cache = TranslationCache::in_memory().unwrap();
        let mut response = google_response("hello friend", Some("es"));
        response.engine = "ct2".to_string();

        cache.insert_google("hola amigo", &response).unwrap();

        assert!(cache.lookup_google("hola amigo", "en").unwrap().is_none());
    }
}
