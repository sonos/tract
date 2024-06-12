#!/bin/sh

set -ex

rustup update

if [ `uname` = "Linux" ]
then
    cargo check --all-targets
else
    cargo check --all-targets --workspace --exclude test-tflite
fi

./.travis/onnx-tests.sh
./.travis/tflite.sh

./.travis/regular-tests.sh
if [ -n "$CI" ]
then
    cargo clean
fi

if [ -n "$CI" ]
then
    cargo clean
fi
./.travis/cli-tests.sh
