#!/bin/sh

export CI=true

set -e

# if [ `uname` = "Linux" -a -z "$TRAVIS" ]
# then
#     apt-get update
#     apt-get -y upgrade
#     apt-get install -y unzip wget curl python awscli build-essential git pkg-config libssl-dev
#     cargo --version || ( curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y )
# fi

. $HOME/.cargo/env

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../.cached
fi

cargo build --release
cargo test --release --all
cargo build --release --benches

if [ -n "$TRAVIS" -a -n "$PARTIAL_CI" ]
then
    exit 0
fi

./.travis/cache_file.sh \
    ARM-ML-KWS-CNN-M.pb \
    GRU128KeywordSpotter-v2-10epochs.onnx \
    inception_v3_2016_08_28_frozen.pb \
    mobilenet_v1_1.0_224_frozen.pb \
    mobilenet_v2_1.4_224_frozen.pb \
    squeezenet.onnx

./target/release/tract $CACHEDIR/squeezenet.onnx \
     run -q --assert-output 1x1000x1x1xf32

./target/release/tract \
     $CACHEDIR/inception_v3_2016_08_28_frozen.pb \
     -i 1x299x299x3xf32 \
     run -q --assert-output-fact 1x1001xf32

./target/release/tract \
    $CACHEDIR/inception_v3_2016_08_28_frozen.pb \
    -i 1x299x299x3xf32 -O \
    run -q --assert-output-fact 1x1001xf32

./target/release/tract $CACHEDIR/ARM-ML-KWS-CNN-M.pb \
     -O -i 49x10xf32 --partial \
     --input-node Mfcc run -q

./target/release/tract $CACHEDIR/mobilenet_v1_1.0_224_frozen.pb \
     -O -i 1x224x224x3xf32 \
    run -q --assert-output-fact 1x1001xf32

./target/release/tract $CACHEDIR/mobilenet_v2_1.4_224_frozen.pb \
     -O -i 1x224x224x3xf32 \
    run -q --assert-output-fact 1x1001xf32

./target/release/tract $CACHEDIR/GRU128KeywordSpotter-v2-10epochs.onnx \
     -O run -q --assert-output-fact 1x3xf32

(
    cd $CACHEDIR
    [ -e librispeech_clean_tdnn_lstm_1e_256.tgz ] \
        || wget -q https://s3.amazonaws.com/tract-ci-builds/fridge/kaldi/librispeech_clean_tdnn_lstm_1e_256.tgz
    [ -d librispeech_clean_tdnn_lstm_1e_256 ] \
        || tar zxf librispeech_clean_tdnn_lstm_1e_256.tgz
)
(
    [ -e kaldi/test_cases/librispeech_clean_tdnn_lstm_1e_256 ] \
        || ln -s `pwd`/$CACHEDIR/librispeech_clean_tdnn_lstm_1e_256 kaldi/test_cases/
    cd kaldi/test_cases
    TRACT_RUN=../../target/release/tract ./run_all.sh
)



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

if [ -z "$TRAVIS" ]
then
    sh .travis/debug-tests.sh
    sh .travis/tf.sh
fi

