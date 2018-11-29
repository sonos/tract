#!/bin/sh

set -ex

export CI=true

# ONNX_CHECKOUT=`pwd`/onnx/.onnx
# TF_INCEPTIONV3=`pwd`/tensorflow/inceptionv3/.inception-v3-2016_08_28
# if [ -n "$TRAVIS" ]
# then
#     ONNX_CHECKOUT=$TRAVIS_BUILD_DIR/cached/onnx-checkout
#     TF_INCEPTIONV3=$TRAVIS_BUILD_DIR/cached/inception-v3-2016_08_28
# fi
# 
# ONNX_TEST_DATA=$ONNX_CHECKOUT/onnx/backend/test/data
# 
# cargo test --release --all --features serialize
# cargo check --benches --all --features serialize # running benches on travis is useless
# 
# cargo check -p tract --features conform
# (cd tensorflow; cargo test --features conform)
# 
# cargo run --release -p tract -- \
#     $ONNX_TEST_DATA/real/test_squeezenet/squeezenet/model.onnx \
#     dump -q --assert-output 1x1000x1x1xf32
# 
# cargo run --release -p tract -- -O \
#     $ONNX_TEST_DATA/real/test_squeezenet/squeezenet/model.onnx \
#     dump -q --assert-output 1x1000x1x1xf32
# 
# cargo run --release -p tract -- \
#     $TF_INCEPTIONV3/inception_v3_2016_08_28_frozen.pb \
#     -i 1x299x299x3xf32 \
#     dump -q --assert-output-fact 1x1001xf32
# 
# cargo run --release -p tract -- \
#     $TF_INCEPTIONV3/inception_v3_2016_08_28_frozen.pb \
#     -i 1x299x299x3xf32 -O \
#     dump -q --assert-output-fact 1x1001xf32

if [ -n "$TRACK_PERF" ]
then
    echo foobar > foobar
    aws s3 cp s3://tract-ci-builds/foobar-$ARCH
fi
