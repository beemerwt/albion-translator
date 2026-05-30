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
