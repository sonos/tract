#!/bin/sh

export CI=true

if [ `uname` = "Darwin" ]
then
    system_profiler SPHardwareDataType
    sysctl -n machdep.cpu.brand_string
fi

set -ex

TF_INCEPTIONV3=`pwd`/tensorflow/inceptionv3/.inception-v3-2016_08_28
if [ -n "$TRAVIS" ]
then
    ONNX_CHECKOUT=$TRAVIS_BUILD_DIR/cached/onnx-checkout
    TF_INCEPTIONV3=$TRAVIS_BUILD_DIR/cached/inception-v3-2016_08_28
fi

cargo check --benches --all --features serialize # running benches on travis is useless
cargo test --release --all --features serialize

(cd tensorflow; cargo test --release --features conform)
(cd cli; cargo build --release)

if [ -z "$ONNX_CHECKOUT" ]
then
    ONNX_CHECKOUT=`find target -name onnx | grep 'out/onnx$' | head -1`
fi

ONNX_TEST_DATA=$ONNX_CHECKOUT/onnx/backend/test/data

./target/release/tract \
    $ONNX_TEST_DATA/real/test_squeezenet/squeezenet/model.onnx \
    dump -q --assert-output 1x1000x1x1xf32

./target/release/tract \
    $ONNX_TEST_DATA/real/test_squeezenet/squeezenet/model.onnx \
    dump -q --assert-output 1x1000x1x1xf32

./target/release/tract \
    $TF_INCEPTIONV3/inception_v3_2016_08_28_frozen.pb \
    -i 1x299x299x3xf32 \
    dump -q --assert-output-fact 1x1001xf32

./target/release/tract \
    $TF_INCEPTIONV3/inception_v3_2016_08_28_frozen.pb \
    -i 1x299x299x3xf32 -O \
    dump -q --assert-output-fact 1x1001xf32
