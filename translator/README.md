# Albion Translator Sidecar

Local FastAPI translation service used by the Rust capture app. It uses Argos
Translate for offline translation behind the stable local REST API.

## Setup

```sh
cd translator
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install test dependencies too:

```sh
python -m pip install -e '.[test]'
```

## Install Argos Packages

Install the requested offline packages outside the Rust build:

```sh
python scripts/install_argos_packages.py
```

The installer attempts:

- `es -> en`
- `pt -> en`
- `zh -> en`
- `vi -> en`

Unavailable language pairs are reported clearly and do not fail the whole run if
at least one requested package is installed or already present.

Argos data, cache, and config files are kept under `translator/.argos/` by
default so the sidecar does not need to write into your global user directories.
The server forces `ARGOS_DEVICE_TYPE=cpu` unless you override it.

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
  -H 'Content-Type: application/json' \
  -d '{"text":"hola mundo","source":"auto","target":"en"}'
```

## Environment

- `ALBION_TRANSLATOR_PORT`: used by the Rust launcher, defaults to `8787`.
- `ALBION_TRANSLATOR_EXTERNAL=1`: Rust will not spawn Python and will only wait
  for `/health`.
- `ALBION_TRANSLATOR_AUTO_INSTALL=1`: the Python server attempts to install
  missing Argos packages at startup. The default is no auto-install.

## Limitations

Argos packages are installed outside the Rust build and may not include every
requested language pair. Vietnamese `vi -> en` may be unavailable depending on
the Argos package index. Language detection uses `langdetect`; short game/chat
messages may be unreliable, so confidence is reported as `null` for short text.

## Tests

The tests do not download models:

```sh
python -m pytest tests
```
