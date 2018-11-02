#!/bin/sh

set -ex

export CI=true

ONNX_CHECKOUT=`pwd`/onnx/.onnx
if [ -n "$TRAVIS" ]
then
    ONNX_CHECKOUT=$TRAVIS_BUILD_DIR/cached/onnx-checkout
fi

ONNX_TEST_DATA=$ONNX_CHECKOUT/onnx/backend/test/data

cargo test --release --all --features serialize
cargo check --benches --all --features serialize # running benches on travis is useless

cargo check -p tract --features conform
cargo test -p tract-tensorflow --features conform

cargo run --release -p tract -- \
    $ONNX_TEST_DATA/real/test_squeezenet/squeezenet/model.onnx \
    dump -q --assert-output 1x1000x1x1xf32

