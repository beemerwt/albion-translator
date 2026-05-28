#!/usr/bin/env bash
set -euo pipefail

export RUST_LOG=1
export ALBION_NETWORK_DEBUG=1

cargo build
sudo setcap cap_net_raw,cap_net_admin+ep ./target/debug/albion-translator
exec ./target/debug/albion-translator --all --pretty "$@"
