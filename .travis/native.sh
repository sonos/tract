#!/bin/sh

set -ex

rustup update

cargo check --all-targets
./.travis/regular-tests.sh

if [ -n "$CI" ]
then
    cargo clean
fi

./.travis/onnx-tests.sh
if [ -n "$CI" ]
then
    cargo clean
fi
./.travis/cli-tests.sh
