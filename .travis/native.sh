#!/bin/sh

set -ex

rustup update

cargo check --all-targets
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
