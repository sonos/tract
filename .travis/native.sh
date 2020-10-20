#!/bin/sh

set -ex

which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y

. $HOME/.cargo/env

: "${RUST_VERSION:=stable}"
rustup toolchain add $RUST_VERSION
rustup default $RUST_VERSION

rustc --version

if [ `uname` = "Darwin" ]
then
    brew install coreutils
fi

# if [ `uname` = "Linux" -a -z "$TRAVIS" ]
# then
#     apt-get update
#     apt-get -y upgrade
#     apt-get install -y unzip wget curl python awscli build-essential git pkg-config libssl-dev
#     cargo --version || ( curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y )
# fi


if [ -z "$CACHEDIR" ]
then
    CACHEDIR=$(realpath `dirname $0`/../.cached)
fi

export CACHEDIR

cargo -q check --workspace --all-targets

if [ `arch` = "x86_64" -a "$RUST_VERSION" = "stable" ]
then
    ALL_FEATURES=--all-features
fi

cargo -q test -q -p tract-core -p tract-hir -p tract-onnx -p tract-linalg
# doc test are not finding libtensorflow.so
cargo -q test -q -p tract-tensorflow --lib $ALL_FEATURES
# useful as debug_asserts will come into play
cargo -q test -q -p onnx-test-suite -- --skip real_ 1_7_0

if [ -n "$SHORT" ]
then
    exit 0
fi

cargo -q test -q -p onnx-test-suite --release

if [ -n "$CI" ]
then
    rm -rf $CACHEDIR/onnx
fi

HARNESS=""
for p in $(ls harness | grep -v onnx-test-suite)
do
    HARNESS="$HARNESS -p $p"
done

cargo -q test -q --release $HARNESS $ALL_FEATURES

cargo -q build -q -p tract --release

./.travis/cache_file.sh \
    ARM-ML-KWS-CNN-M.pb \
    GRU128KeywordSpotter-v2-10epochs.onnx \
    hey_snips_v4_model17.pb \
    hey_snips_v4_model17.alpha1.tgz \
    inception_v3_2016_08_28_frozen.pb \
    mdl-en-2019-Q3-librispeech.onnx \
    mobilenet_v1_1.0_224_frozen.pb \
    mobilenet_v2_1.4_224_frozen.pb \
    squeezenet.onnx \
    en_libri_real.tar.gz

(
cd $CACHEDIR
[ -d en_libri_real ] || tar zxf en_libri_real.tar.gz
)

./target/release/tract $CACHEDIR/squeezenet.onnx \
    run -q --assert-output 1x1000x1x1xf32

./target/release/tract \
    $CACHEDIR/inception_v3_2016_08_28_frozen.pb \
    -i 1,299,299,3,f32 \
    run -q --assert-output-fact 1x1001xf32

./target/release/tract \
    $CACHEDIR/inception_v3_2016_08_28_frozen.pb \
    -i 1,299,299,3,f32 -O \
    run -q --assert-output-fact 1x1001xf32

./target/release/tract $CACHEDIR/ARM-ML-KWS-CNN-M.pb \
    -O -i 49,10,f32 --partial \
    --input-node Mfcc run -q

./target/release/tract $CACHEDIR/mdl-en-2019-Q3-librispeech.onnx \
    -O -i S,40,f32 --output-node output --pulse 24 \
    run -q

./target/release/tract $CACHEDIR/mobilenet_v1_1.0_224_frozen.pb \
    -O -i 1,224,224,3,f32 \
    run -q --assert-output-fact 1x1001xf32

./target/release/tract $CACHEDIR/mobilenet_v2_1.4_224_frozen.pb \
    -O -i 1,224,224,3,f32 \
    run -q --assert-output-fact 1x1001xf32

./target/release/tract $CACHEDIR/GRU128KeywordSpotter-v2-10epochs.onnx \
    -O run -q --assert-output-fact 1,3,f32

./target/release/tract $CACHEDIR/hey_snips_v4_model17.pb \
    -i S,20,f32 --pulse 8 dump --cost -q \
    --assert-cost "FMA(F32)=2060448,Div(F32)=24576,Buffer(F32)=2920,Params(F32)=222250"

# fragile test (generated names...) but kinda vital for AM perf
./target/release/tract $CACHEDIR/mdl-en-2019-Q3-librispeech.onnx \
    -i S,40 --output-node output dump \
    --node-name "fastlstm1.c_final.extracted.fastlstm1.four_parts" \
    | grep -c MatMul | grep 4

./target/release/tract $CACHEDIR/mdl-en-2019-Q3-librispeech.onnx \
    -i S,40 --output-node output dump \
    --node-name "fastlstm2.c_final.extracted.fastlstm2.four_parts" \
    | grep -c MatMul | grep 4

[ ! -z "$(./target/release/tract $CACHEDIR/hey_snips_v4_model17.pb -i S,20,f32 --pass type dump --op-name AddAxis)" ]
[ -z "$(./target/release/tract $CACHEDIR/hey_snips_v4_model17.pb -i S,20,f32 dump --op-name AddAxis)" ]

( cd kaldi/test_cases ; TRACT_RUN=../../target/release/tract ./run_all.sh )
( cd onnx/test_cases ; TRACT_RUN=../../target/release/tract ./run_all.sh )

#./target/release/tract $CACHEDIR/en_libri_real/model.raw.txt \
#    -f kaldi --output-node output \
#    --kaldi-downsample 3 --kaldi-left-context 5 --kaldi-right-context 15 --kaldi-adjust-final-offset -5 \
#    -i Sx40 --pulse 24 cost -q \
#    --assert-cost "FMA(F32)=23201280,Div(F32)=20480,Buffer(F32)=1896"

./target/release/tract $CACHEDIR/en_libri_real/model.raw.txt \
    -f kaldi --output-node output \
    --kaldi-downsample 3 --kaldi-left-context 5 --kaldi-right-context 15 --kaldi-adjust-final-offset -5 \
    --input-bundle $CACHEDIR/en_libri_real/io.npz \
    run \
    --assert-output-bundle $CACHEDIR/en_libri_real/io.npz

./target/release/tract $CACHEDIR/en_libri_real/model.raw \
    -f kaldi --output-node output \
    --kaldi-downsample 3 --kaldi-left-context 5 --kaldi-right-context 15 --kaldi-adjust-final-offset -5 \
    --input-bundle $CACHEDIR/en_libri_real/io.npz \
    run \
    --assert-output-bundle $CACHEDIR/en_libri_real/io.npz

./target/release/tract $CACHEDIR/en_libri_real/model.onnx \
    --output-node output \
    --kaldi-left-context 5 --kaldi-right-context 15 --kaldi-adjust-final-offset -5 \
    --input-bundle $CACHEDIR/en_libri_real/io.npz \
    run \
    --assert-output-bundle $CACHEDIR/en_libri_real/io.npz

# these tests require access to private snips models
if [ -e "$HOME/.aws/credentials" ]
then
    BENCH_OPTS="--max-iters 1" sh .travis/bundle-entrypoint.sh
    (
    cd onnx/test_cases
    [ -e en_tdnn_lstm_bn_q7 ] || ln -s "$CACHEDIR/en_tdnn_lstm_bn_q7" .
    TRACT_RUN=../../target/release/tract ./run_all.sh en_tdnn_lstm_bn_q7
)
fi

