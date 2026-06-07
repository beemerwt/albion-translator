#!/usr/bin/env bash
set -euo pipefail

export RUST_LOG=1
export ALBION_NETWORK_DEBUG=1
export TRANSLATION_MODEL_DIR=./models-cache

cargo build --no-default-features --features translation-ct2-cuda
sudo setcap cap_net_raw,cap_net_admin+ep ./target/debug/albion-translator
exec ./target/debug/albion-translator --pretty "$@"
