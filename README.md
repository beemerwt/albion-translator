# Albion Translator

Live-capture scaffold for decoding Albion Online Photon UDP traffic with the local
`albion-network-lib` crate at `../albion-network-lib`.

## Requirements

- Rust 2024 toolchain
- `libpcap` development headers
- Capture privileges for network interfaces
- Python 3.11+ for the local translation sidecar

On Linux, either run with `sudo` or grant capture capabilities to your built
binary after compiling.

## Usage

Capture from all openable Ethernet interfaces:

```sh
cargo run
```

Capture always uses UDP port `5056` and filters decoded packets to endpoints
listed in `hosts.txt`. Add Albion IP ranges there in CIDR format, one per line.

Only decoded packets with extracted Albion models are printed by default as JSON
Lines. To print every decoded Photon packet, pass `--all`.

```sh
cargo run -- --all
```

Useful options:

```sh
cargo run -- --pretty
cargo run -- --debug --all
```

## Native Translation Backend

The Rust app now routes translation through an engine-neutral Rust abstraction.
The native path uses CTranslate2 through `ct2rs` and does not call Python or
Argos. Argos is still available as a deprecated fallback while the ct2 model
path is rolled out.

Backend selection:

```sh
TRANSLATION_BACKEND=auto cargo run
TRANSLATION_BACKEND=ct2 cargo run
TRANSLATION_BACKEND=argos cargo run
TRANSLATION_BACKEND=google cargo run
TRANSLATION_BACKEND=http TRANSLATION_HTTP_CONFIG=presets/translategemma-vllm.json cargo run
TRANSLATION_BACKEND=translategemma-vllm cargo run
TRANSLATION_BACKEND=noop cargo run
```

`auto` prefers ct2 when a supported model is installed. If no ct2 model is
available, it falls back to the deprecated Argos sidecar unless disabled:

```sh
ALBION_TRANSLATOR_ARGOS_FALLBACK=0 cargo run
```

Source-language routing:

```sh
TRANSLATION_SOURCE_LANG=auto cargo run
TRANSLATION_SOURCE_LANG=es cargo run
TRANSLATION_SOURCE_LANG=pt cargo run
TRANSLATION_TARGET_LANG=en cargo run
TRANSLATION_DETECTION_CONFIDENCE_THRESHOLD=0.65 cargo run
```

`auto` detects one source language locally with `lingua`, then routes only to
the matching installed `source -> target` model. Low-confidence detections,
very short messages, URLs, number-only text, emoji/punctuation-only text, and
already-target-language messages are skipped without warning spam. Manual source
values are useful for debugging and bypass automatic detection.

Google Translate is available as a remote backend:

```sh
cp .env.example .env
TRANSLATION_BACKEND=google cargo run
```

Fill in `GOOGLE_TRANSLATE_API_KEY` in `.env`. The app loads `.env` at startup
with `dotenvy`, and shell environment variables still take precedence.
`TRANSLATION_GOOGLE_API_KEY` is also accepted. Google Translate always uses
Google's own source-language auto-detection; local `lingua` detection is only
used for skip/routing decisions and as a fallback if Google omits a detected
source language.

Custom HTTP translation backends are configured with JSON templates:

```sh
TRANSLATION_BACKEND=http \
TRANSLATION_HTTP_CONFIG=presets/translategemma-vllm.json \
cargo run
```

The built-in TranslateGemma vLLM preset can also be selected directly:

```sh
TRANSLATION_BACKEND=translategemma-vllm cargo run
```

The preset targets a local vLLM OpenAI-compatible completions server at
`http://localhost:8000/v1/completions`, model
`Infomaniak-AI/vllm-translategemma-4b-it`, and this prompt format:

```text
<<<source>>>{src_lang}<<<target>>>{target_lang}<<<text>>>{text}<<</text>>>
```

HTTP template config files support `endpoint`, `method`, interpolated
`headers`, interpolated JSON `body`, response extraction with paths such as
`$.choices[0].text`, `$.choices[0].message.content`, `$.response`, and
`$.translated_text`, plus output cleanup rules. Available placeholders are
`{text}`, `{src_lang}`, `{target_lang}`, and optional `{api_key}`. Set
`TRANSLATION_HTTP_API_KEY` when a header such as `Authorization: Bearer
{api_key}` is needed; headers containing `{api_key}` are omitted when no key is
configured.

Translated chat output includes the detected source language:

```text
[4:32 PM][Say][es] PlayerName: hello
```

Google translations are cached in SQLite before any backend is queried. The
default database is `./translations.sqlite3`; override it with
`TRANSLATION_CACHE_DB=/path/to/translations.sqlite3`. Cache keys use trimmed
text with repeated whitespace collapsed, plus target language. Only Google
results are inserted into the SQLite cache for now.

Runtime device selection for ct2:

```sh
TRANSLATION_DEVICE=cpu cargo run --features translation-ct2-cpu
TRANSLATION_DEVICE=cuda cargo run --features translation-ct2-cuda
TRANSLATION_DEVICE=auto cargo run --features translation-ct2-cuda
```

`cpu` always uses CPU. `cuda` requires a CUDA-enabled build and fails clearly if
GPU support cannot be loaded, unless `TRANSLATION_ALLOW_DEVICE_FALLBACK=1` is
set. `auto` prefers CUDA in a CUDA build and otherwise uses CPU. CPU and CUDA
artifacts should be built separately for Linux packaging:

```sh
cargo build --release --features translation-ct2-cpu
cargo build --release --no-default-features --features translation-ct2-cuda
```

For Rust-only tests that do not require native CTranslate2 tooling:

```sh
cargo test --no-default-features
```

Linux native dependencies for ct2 builds include a C/C++ toolchain, `cmake`,
and the native libraries required by the selected `ct2rs` features. CUDA builds
also require the NVIDIA driver, CUDA runtime/toolkit, and CTranslate2 CUDA
library support. Do not vendor large CUDA libraries into this repository; use a
package step or dynamic linker path that places the expected shared libraries
beside the binary or in the system library path.

### Models

Model metadata lives in `models/manifest.json`; converted model assets do not
belong in git. The initial manifest declares `es -> en` as
`opus-mt-es-en-ct2`.

The runtime searches for models in this order:

1. `TRANSLATION_MODEL_DIR` or `ALBION_TRANSLATION_MODEL_DIR`
2. bundled `models/` directory beside the executable
3. project-local `models-cache/`
4. user cache/data locations under `~/.cache/albion-translator/models` and
   `~/.local/share/albion-translator/models`

Expected layout:

```text
models-cache/
  opus-mt-es-en-int8/
    model.bin
    config.json
    source.spm
    target.spm
```

Use a custom manifest when needed:

```sh
TRANSLATION_MODEL_MANIFEST=/path/to/manifest.json \
TRANSLATION_MODEL_DIR=/path/to/models \
cargo run --features translation-ct2-cpu
```

`build.rs` reads the manifest and warns when local development models are
missing. It does not download, prepare, or convert model files during normal
builds. Future download support is reserved for:

```sh
DOWNLOAD_TRANSLATION_MODELS=1 cargo build --features translation-ct2-cpu
REQUIRE_TRANSLATION_MODELS=1 cargo build --features translation-ct2-cpu
```

Run ignored smoke tests after installing a local model:

```sh
TRANSLATION_MODEL_DIR=models-cache cargo test --features translation-ct2-cpu -- --ignored smoke_translate_spanish_to_english
TRANSLATION_MODEL_DIR=models-cache cargo test --no-default-features --features translation-ct2-cuda -- --ignored smoke_translate_spanish_to_english_cuda
```

### Translation model preparation

End users do not run the model-preparation scripts. They are for developers and
release builders assembling official release artifacts. Official releases should
ship with a prepared `models/` directory, so users do not need Python, Hugging
Face tooling, internet access, or conversion steps.

The source metadata is `models/manifest.json`. Converted model binaries are
written under `target/translation-models/` by default and must not be committed
to git.

Install the release-builder Python tools:

```sh
python -m venv .venv-models
source .venv-models/bin/activate
pip install --upgrade pip
pip install ctranslate2 transformers sentencepiece huggingface_hub
```

Prepare the initial Spanish-to-English model:

```sh
python scripts/prepare_models.py --model es-en --quantization int8
```

The script is idempotent: if the converted model directory already contains
`config.json` and `model*.bin`, it skips conversion. Force regeneration or preview
the work with:

```sh
python scripts/prepare_models.py --model es-en --quantization int8 --force
python scripts/prepare_models.py --model es-en --quantization int8 --dry-run
```

Prepare every model declared in the manifest:

```sh
python scripts/prepare_models.py --all --quantization int8
```

By default the prepared tree looks like:

```text
target/translation-models/
  manifest.json
  opus-mt-es-en-int8/
    config.json
    model.bin
    source.spm
    target.spm
    tokenizer_config.json
    special_tokens_map.json
```

### Linux release packaging

`scripts/package_release.py` assembles release folders from an already-built Rust
binary and already-prepared model assets. It does not build Rust, download
models, or run conversion.

CPU release flow:

```sh
python -m venv .venv-models
source .venv-models/bin/activate
pip install --upgrade pip
pip install ctranslate2 transformers sentencepiece huggingface_hub

python scripts/prepare_models.py --model es-en --quantization int8
cargo build --release --features translation-ct2-cpu
python scripts/package_release.py --target linux-cpu
```

CUDA release flow:

```sh
python -m venv .venv-models
source .venv-models/bin/activate
pip install --upgrade pip
pip install ctranslate2 transformers sentencepiece huggingface_hub

python scripts/prepare_models.py --model es-en --quantization int8
cargo build --release --no-default-features --features translation-ct2-cuda
python scripts/package_release.py --target linux-cuda
```

The release layout is:

```text
dist/
  linux-cpu/
    albion-translator
    models/
      manifest.json
      opus-mt-es-en-int8/

  linux-cuda/
    albion-translator
    CUDA_SHARED_LIBS_TODO.txt
    models/
      manifest.json
      opus-mt-es-en-int8/
```

Use `--binary` or `--dist-dir` to override paths. Existing
release directories are not overwritten unless `--force` is provided, and
`--dry-run` prints the planned copies without writing files.

## Deprecated Translation Sidecar

The Rust app starts a local FastAPI translation sidecar on `127.0.0.1` when it
starts, waits for `GET /health`, and kills the child process when the Rust app
exits. It prefers `translator/.venv/bin/python` when that virtual environment
exists, then falls back to `python3`. The Rust code is synchronous, so the client
uses `reqwest`'s blocking API instead of adding a Tokio runtime.

This Argos path is deprecated and should only be used as a migration fallback.

The default sidecar port is `8787`:

```sh
ALBION_TRANSLATOR_PORT=8787 cargo run
```

To use a manually started sidecar instead of letting Rust spawn Python:

```sh
ALBION_TRANSLATOR_EXTERNAL=1 cargo run
```

Create the Python environment:

```sh
cd translator
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the sidecar manually:

```sh
cd translator
. .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
```

Install Argos offline translation packages:

```sh
cd translator
. .venv/bin/activate
python scripts/install_argos_packages.py
```

Optional Python startup auto-install:

```sh
ALBION_TRANSLATOR_AUTO_INSTALL=1 cargo run
```

Argos models are installed outside the Rust build and may not include every
requested language pair. Vietnamese `vi -> en` may be unavailable depending on
the Argos package index. Language detection uses `langdetect`; short game/chat
messages may be unreliable.
