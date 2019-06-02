#!/bin/sh

export CI=true

set -ex

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../.cached
fi

cargo build --release
cargo test --release --all
cargo build --release --benches

if [ -z "$TRAVIS" ]
then
    sh .travis/debug-tests.sh
    sh .travis/tf.sh
fi


./.travis/cache_file.sh \
    ARM-ML-KWS-CNN-M.pb \
    inception_v3_2016_08_28_frozen.pb \
    mobilenet_v1_1.0_224_frozen.pb \
    mobilenet_v2_1.4_224_frozen.pb \
    squeezenet.onnx

./target/release/tract $CACHEDIR/squeezenet.onnx \
     dump -q --assert-output 1x1000x1x1xf32

./target/release/tract \
     $CACHEDIR/inception_v3_2016_08_28_frozen.pb \
     -i 1x299x299x3xf32 \
     dump -q --assert-output-fact 1x1001xf32

./target/release/tract \
    $CACHEDIR/inception_v3_2016_08_28_frozen.pb \
    -i 1x299x299x3xf32 -O \
    dump -q --assert-output-fact 1x1001xf32

 ./target/release/tract $CACHEDIR/ARM-ML-KWS-CNN-M.pb \
     -O -i 49x10xf32 \
     --input-node Mfcc run > /dev/null

 ./target/release/tract $CACHEDIR/mobilenet_v1_1.0_224_frozen.pb \
     -O -i 1x224x224x3xf32 \
    dump -q --assert-output-fact 1x1001xf32

 ./target/release/tract $CACHEDIR/mobilenet_v2_1.4_224_frozen.pb \
     -O -i 1x224x224x3xf32 \
    dump -q --assert-output-fact 1x1001xf32

# these tests require access to private snips models
if [ -n "$RUN_ALL_TESTS" ]
then
    for model in \
        snips-voice-commands-cnn-float.pb \
        snips-voice-commands-cnn-fake-quant.pb
    do
        (cd $CACHEDIR/ ; [ -e $model ] || 
            aws s3 cp s3://tract-ci-builds/tests/snips/$model . )
    done

    ./target/release/tract $CACHEDIR/snips-voice-commands-cnn-float.pb \
        -O -i 200x10xf32 run > /dev/null

    ./target/release/tract $CACHEDIR/snips-voice-commands-cnn-fake-quant.pb \
        -O -i 200x10xf32 run > /dev/null

fi
