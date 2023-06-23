#!/bin/sh

set -ex

rustup update

cargo check --all-targets

./.travis/regular-tests.sh

cargo test -q -p tract-core --features paranoid_assertions

# useful as debug_asserts will come into play
cargo -q test -q -p test-onnx-core
cargo -q test -q -p test-onnx-nnef-cycle

cargo check -p tract-nnef --features complex
cargo check -p tract-tflite
cargo check -p tract --no-default-features

./.travis/onnx-tests.sh
./.travis/cli-tests.sh
