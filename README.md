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

## Translation Sidecar

The Rust app starts a local FastAPI translation sidecar on `127.0.0.1` when it
starts, waits for `GET /health`, and kills the child process when the Rust app
exits. It prefers `translator/.venv/bin/python` when that virtual environment
exists, then falls back to `python3`. The Rust code is synchronous, so the client
uses `reqwest`'s blocking API instead of adding a Tokio runtime.

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

Current limitation: translation and language detection are stubbed. The REST
contract is stable, but no real offline ML model is wired in yet. Future backend
options include LibreTranslate/Argos Translate style backends, NLLB-200 via
transformers, and OPUS-MT/Helsinki-NLP models.
