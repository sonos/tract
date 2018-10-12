#!/bin/sh

set -ex

export CI=true

ONNX_CHECKOUT=`pwd`/tfdeploy-onnx/.onnx
if [ -n "$TRAVIS" ]
then
    ONNX_CHECKOUT=$TRAVIS_BUILD_DIR/cached/onnx-checkout
fi

ONNX_TEST_DATA=$ONNX_CHECKOUT/onnx/backend/test/data

cargo build --release
cargo test --release --all 
cargo check --benches --all # running benches on travis is useless

cargo run --release -p cli -- \
    $ONNX_TEST_DATA/real/test_squeezenet/squeezenet/model.onnx \
    dump -q --assert-output 1x1000x1x1xf32
