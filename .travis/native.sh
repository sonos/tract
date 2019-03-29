#!/bin/sh

export CI=true

set -ex

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../.cached
fi

cargo check --benches --all --features serialize # running benches on travis is useless
cargo test --release --all --features serialize

(cd tensorflow; cargo test --release --features conform)
(cd examples/tf-inceptionv3; cargo test --release)
(cd examples/lstm-proptest-onnx-vs-tf; cargo test --release)
(cd examples/tf-moz-deepspeech ; cargo test --release)
(cd cli; cargo build --release)

./.travis/cache_file.sh \
    ARM-ML-KWS-CNN-M.pb \
    inception_v3_2016_08_28_frozen.pb \
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
