#!/bin/sh

export CI=true

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
    ONNX_CHECKOUT=target
fi

./target/release/tract \
    `find $ONNX_CHECKOUT -name model.onnx | grep squeezenet` \
    dump -q --assert-output 1x1000x1x1xf32

./target/release/tract \
    $TF_INCEPTIONV3/inception_v3_2016_08_28_frozen.pb \
    -i 1x299x299x3xf32 \
    dump -q --assert-output-fact 1x1001xf32

./target/release/tract \
    $TF_INCEPTIONV3/inception_v3_2016_08_28_frozen.pb \
    -i 1x299x299x3xf32 -O \
    dump -q --assert-output-fact 1x1001xf32

CACHEDIR=${CACHEDIR:-$HOME/.cache}

if [ -n "$RUN_ALL_TEST" -a -n "$AWS_ACCESS_KEY_ID" -a -e "target/$RUSTC_TRIPLE/release/tract" ]
then
    (cd $CACHEDIR ; aws s3 sync s3://tract-ci-builds/model $CACHEDIR)
    ./target/release/tract $CACHEDIR/ARM-ML-KWS-CNN-M.pb -O -i 49x10xf32 \
        --input-node Mfcc run
fi
