#!/usr/bin/env bash
set -euo pipefail

# Force CMake/NVCC to use GCC 15 instead of Fedora's default GCC 16
export CC=/usr/bin/gcc-15
export CXX=/usr/bin/g++-15
export CUDAHOSTCXX=/usr/bin/g++-15

export CUDA_ARCH_LIST=120
export RUST_LOG=1
export ALBION_NETWORK_DEBUG=1
export TRANSLATION_MODEL_DIR=./models-cache

cargo clean -p ct2rs
cargo build --no-default-features --features translation-ct2-cuda

sudo setcap cap_net_raw,cap_net_admin+ep ./target/debug/albion-translator
exec ./target/debug/albion-translator --pretty "$@"
