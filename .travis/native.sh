#!/bin/sh

set -ex

rustup update

cargo check --all-targets --workspace --exclude test-tflite

./.travis/onnx-tests.sh
./.travis/regular-tests.sh

if [ -n "$CI" ]
then
    cargo clean
fi

if [ `uname` = "Linux" ]
then
    ./.travis/tflite.sh
fi

if [ -n "$CI" ]
then
    cargo clean
fi
./.travis/cli-tests.sh
