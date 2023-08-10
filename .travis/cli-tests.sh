#!/bin/sh

WHITE='\033[1;37m'
NC='\033[0m' # No Color

set -e

which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y
rustup update

PATH=$PATH:$HOME/.cargo/bin

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

echo
echo $WHITE • build tract $NC
echo

TRACT_RUN=$(cargo build --message-format json -p tract $CARGO_EXTRA --profile opt-no-lto | jq -r 'select(.target.name == "tract" and .executable).executable')
echo TRACT_RUN=$TRACT_RUN
export TRACT_RUN

echo
echo $WHITE • harness/nnef-test-cases $NC
echo

for t in `find harness/nnef-test-cases -name runme.sh`
do
    echo $WHITE$t$NC
    $t
done

echo
echo $WHITE • onnx/test_cases $NC
echo

MODELS=https://s3.amazonaws.com/tract-ci-builds/tests

( cd onnx/test_cases ; CACHEDIR=$MODELS ./run_all.sh )

echo
echo $WHITE • old integration test cases $NC
echo

$TRACT_RUN $MODELS/squeezenet.onnx \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1000,1,1,f32

$TRACT_RUN \
    $MODELS/inception_v3_2016_08_28_frozen.pb \
    -i 1,299,299,3,f32 \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1001,f32

$TRACT_RUN \
    $MODELS/inception_v3_2016_08_28_frozen.pb \
    -i 1,299,299,3,f32 -O \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1001,f32

$TRACT_RUN $MODELS/ARM-ML-KWS-CNN-M.pb \
    -O -i 49,10,f32 --partial \
    --input-node Mfcc \
    run -q \
    --allow-random-input
    
$TRACT_RUN $MODELS/mdl-en-2019-Q3-librispeech.onnx \
    -O -i S,40,f32 --output-node output --pulse 24 \
    run -q \
    --allow-random-input
    
$TRACT_RUN $MODELS/mobilenet_v1_1.0_224_frozen.pb \
    -O -i 1,224,224,3,f32 \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1001,f32

$TRACT_RUN $MODELS/mobilenet_v2_1.4_224_frozen.pb \
    -O -i 1,224,224,3,f32 \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1001,f32

$TRACT_RUN $MODELS/GRU128KeywordSpotter-v2-10epochs.onnx \
    -O run -q \
    --allow-random-input \
    --assert-output-fact 1,3,f32

$TRACT_RUN $MODELS/hey_snips_v4_model17.pb \
    -i S,20,f32 --pulse 8 dump --cost -q \
    --assert-cost "FMA(F32)=2060448,Div(F32)=24576,Buffer(F32)=2920,Params(F32)=222250"

$TRACT_RUN $MODELS/hey_snips_v4_model17.pb -i S,20,f32 \
    dump -q \
    --assert-op-count AddAxis 0

$TRACT_RUN $MODELS/en_libri_real/model.onnx \
    --output-node output \
    --edge-left-context 5 --edge-right-context 15 \
    --input-facts-from-bundle $MODELS/en_libri_real/io.npz \
    run \
    --input-from-bundle $MODELS/en_libri_real/io.npz \
    --allow-random-input \
    --assert-output-bundle $MODELS/en_libri_real/io.npz

$TRACT_RUN $MODELS/inceptionv1_quant.nnef.tar.gz \
    --nnef-tract-core \
    --input-facts-from-bundle $MODELS/inceptionv1_quant.io.npz \
    run \
    --input-from-bundle $MODELS/inceptionv1_quant.io.npz \
    --allow-random-input \
    --assert-output-bundle $MODELS/inceptionv1_quant.io.npz

for t in harness/pre-optimized-graphes/*
do
    ( cd $t ; CACHEDIR=$MODELS ./runme.sh)
done

(
if aws s3 ls tract-ci-builds/model/private
then
    echo
    echo $WHITE • private tests $NC
    echo
    if [ -n "$CI" ]
    then
        OUTPUT=/dev/null
    else
        set -x
        OUTPUT=/dev/stdout
    fi
    (
    mkdir -p $CACHEDIR
    cd $CACHEDIR
    aws s3 sync s3://tract-ci-builds/model/private private
    for t in `find private -name t.sh`
    do
        ( cd `dirname $t` ; sh ./t.sh )
    done
    ) 2>&1 > $OUTPUT

    echo
    echo $WHITE • benches on full models $NC
    echo

    ./.travis/bundle-entrypoint.sh
fi
)

