use anyhow::{Result, anyhow};
use std::{
    collections::{HashMap, VecDeque},
    num::NonZeroUsize,
    sync::Mutex,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslateRequest {
    pub source_lang: String,
    pub target_lang: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslateResponse {
    pub translated_text: String,
    pub engine: String,
    pub model_id: Option<String>,
    pub device: String,
}

pub trait Translator: Send + Sync {
    fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse>;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub source_lang: String,
    pub target_lang: String,
    pub text: String,
    pub model_id: Option<String>,
}

impl CacheKey {
    pub fn new(request: &TranslateRequest, model_id: Option<String>) -> Self {
        Self {
            source_lang: request.source_lang.clone(),
            target_lang: request.target_lang.clone(),
            text: request.text.clone(),
            model_id,
        }
    }
}

pub struct CachedTranslator<T> {
    inner: T,
    model_id: Option<String>,
    cache: Mutex<TranslationCache>,
}

impl<T> CachedTranslator<T> {
    pub fn new(inner: T, model_id: Option<String>, capacity: usize) -> Self {
        Self {
            inner,
            model_id,
            cache: Mutex::new(TranslationCache::new(capacity)),
        }
    }
}

impl<T: Translator> Translator for CachedTranslator<T> {
    fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse> {
        let key = CacheKey::new(&request, self.model_id.clone());
        if let Some(response) = self
            .cache
            .lock()
            .map_err(|_| anyhow!("translation cache lock was poisoned"))?
            .get(&key)
        {
            return Ok(response);
        }

        let response = self.inner.translate(request)?;
        self.cache
            .lock()
            .map_err(|_| anyhow!("translation cache lock was poisoned"))?
            .insert(key, response.clone());
        Ok(response)
    }
}

#[derive(Debug)]
struct TranslationCache {
    capacity: Option<NonZeroUsize>,
    values: HashMap<CacheKey, TranslateResponse>,
    order: VecDeque<CacheKey>,
}

impl TranslationCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: NonZeroUsize::new(capacity),
            values: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<TranslateResponse> {
        let value = self.values.get(key)?.clone();
        self.touch(key);
        Some(value)
    }

    fn insert(&mut self, key: CacheKey, response: TranslateResponse) {
        let Some(capacity) = self.capacity else {
            return;
        };

        if self.values.contains_key(&key) {
            self.values.insert(key.clone(), response);
            self.touch(&key);
            return;
        }

        while self.values.len() >= capacity.get() {
            if let Some(oldest) = self.order.pop_front() {
                self.values.remove(&oldest);
            }
        }

        self.order.push_back(key.clone());
        self.values.insert(key, response);
    }

    fn touch(&mut self, key: &CacheKey) {
        self.order.retain(|candidate| candidate != key);
        self.order.push_back(key.clone());
    }
}

#[derive(Debug, Clone)]
pub struct NoopTranslator;

impl Translator for NoopTranslator {
    fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse> {
        Ok(TranslateResponse {
            translated_text: request.text,
            engine: "noop".to_string(),
            model_id: None,
            device: "none".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Clone)]
    struct CountingTranslator {
        calls: Arc<AtomicUsize>,
        model_id: Option<String>,
    }

    impl Translator for CountingTranslator {
        fn translate(&self, request: TranslateRequest) -> Result<TranslateResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(TranslateResponse {
                translated_text: format!("translated: {}", request.text),
                engine: "test".to_string(),
                model_id: self.model_id.clone(),
                device: "cpu".to_string(),
            })
        }
    }

    fn request(text: &str) -> TranslateRequest {
        TranslateRequest {
            source_lang: "es".to_string(),
            target_lang: "en".to_string(),
            text: text.to_string(),
        }
    }

    #[test]
    fn cache_key_includes_model_id() {
        let first = CacheKey::new(&request("hola"), Some("model-a".to_string()));
        let second = CacheKey::new(&request("hola"), Some("model-b".to_string()));

        assert_ne!(first, second);
    }

    #[test]
    fn cached_translator_reuses_repeated_message() {
        let calls = Arc::new(AtomicUsize::new(0));
        let translator = CachedTranslator::new(
            CountingTranslator {
                calls: calls.clone(),
                model_id: Some("es-en-test".to_string()),
            },
            Some("es-en-test".to_string()),
            8,
        );

        let first = translator.translate(request("hola")).unwrap();
        let second = translator.translate(request("hola")).unwrap();

        assert_eq!(first, second);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn noop_translator_returns_original_text() {
        let response = NoopTranslator.translate(request("hola")).unwrap();

        assert_eq!(response.translated_text, "hola");
        assert_eq!(response.engine, "noop");
        assert_eq!(response.device, "none");
    }
}
