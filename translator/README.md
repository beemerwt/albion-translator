# Albion Translator Sidecar

Local FastAPI translation service used by the Rust capture app.

The current implementation is intentionally a stub. It keeps the REST API stable
without downloading or bundling large machine translation models.

## Setup

```sh
cd translator
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run Manually

```sh
cd translator
. .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
```

Health check:

```sh
curl http://127.0.0.1:8787/health
```

Translate:

```sh
curl -X POST http://127.0.0.1:8787/translate \
  -H 'content-type: application/json' \
  -d '{"text":"hola mundo","source":"auto","target":"en"}'
```

## Limitations

Language detection and translation are placeholders. Responses are marked as
stub output and should not be treated as accurate.

Future offline model backends could include:

- LibreTranslate/Argos Translate style backend
- NLLB-200 via transformers
- OPUS-MT/Helsinki-NLP models
