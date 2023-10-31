#!/bin/sh

set -ex

rustup update

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
