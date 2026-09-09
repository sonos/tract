#!/bin/sh

set -ex

# Feature combinations no other CI job exercises. Downstream libraries link
# subsets of tract (typically NNEF loading plus the pulse runtime, no CLI, no
# ONNX, no GPU), and nothing kept those subsets compiling.

check() {
    cargo check -p "$@" $CARGO_EXTRA
}

check tract-data --no-default-features
check tract-data --features complex

check tract-core --no-default-features
check tract-core --features complex
check tract-core --features paranoid_assertions

check tract-linalg --no-default-features
check tract-linalg --features no_fp16
check tract-linalg --features foreign-inventory

check tract-nnef --no-default-features
check tract-nnef --features complex
check tract-nnef --features unstable-jinja
check tract-nnef --no-default-features --features complex

check tract-pulse-opl --no-default-features
check tract-pulse-opl --features complex

check tract-pulse --no-default-features
check tract-transformers --no-default-features
check tract-extra --no-default-features
check tract-onnx --no-default-features
check tract-onnx-opl --no-default-features

check tract-libcli --no-default-features
check tract-libcli --features onnx,transformers

check tract --no-default-features
check tract-cli --no-default-features

check tract --no-default-features --features flate2

check tract --no-default-features --features pulse

check tract-nnef --features unstable-safetensors

check tract --no-default-features --features transformers

check tract-onnx --features transformers
