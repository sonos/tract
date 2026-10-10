#!/bin/sh

set -ex

rustup update

cargo update
cargo check --all-targets --workspace --exclude test-tflite --exclude test-metal --exclude tract-metal

./ci/onnx-tests.sh
./ci/regular-tests.sh
./ci/test-harness.sh

if [ -n "$CI" ]
then
    cargo clean
fi

if [ `uname` = "Linux" ]
then
    ./ci/tflite.sh
fi

if [ -n "$CI" ]
then
    cargo clean
fi
if nvidia-smi > /dev/null 2>&1
then
    cargo test -p tract-cuda --lib
    cargo test -p test-cuda
fi

./ci/cli-tests.sh
