#!/bin/sh

WHITE='\033[1;37m'
NC='\033[0m' # No Color

set -e

ROOT=$(dirname $(dirname $(realpath $0)))
. $ROOT/.travis/ci-system-setup.sh

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

# ( cd onnx/test_cases ; CACHEDIR=$MODELS ./run_all.sh )

echo
echo $WHITE • full models command line test cases $NC
echo

echo $WHITE     image $NC

$CACHE_FILE squeezenet.onnx
$TRACT_RUN $MODELS/squeezenet.onnx -O \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1000,1,1,f32

$CACHE_FILE inception_v3_2016_08_28_frozen.pb
$TRACT_RUN \
    $MODELS/inception_v3_2016_08_28_frozen.pb \
    -i 1,299,299,3,f32 -O \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1001,f32

$TRACT_RUN \
    $MODELS/inception_v3_2016_08_28_frozen.pb \
    -i 1,299,299,3,f32 -O \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1001,f32

$CACHE_FILE mobilenet_v1_1.0_224_frozen.pb
$TRACT_RUN $MODELS/mobilenet_v1_1.0_224_frozen.pb \
    -O -i 1,224,224,3,f32 \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1001,f32

$CACHE_FILE $MODELS/mobilenet_v2_1.4_224_frozen.pb
$TRACT_RUN $MODELS/mobilenet_v2_1.4_224_frozen.pb \
    -O -i 1,224,224,3,f32 \
    run -q \
    --allow-random-input \
    --assert-output-fact 1,1001,f32

$CACHE_FILE inceptionv1_quant.nnef.tar.gz inceptionv1_quant.io.npz
$TRACT_RUN $MODELS/inceptionv1_quant.nnef.tar.gz \
    --nnef-tract-core \
    --input-facts-from-bundle $MODELS/inceptionv1_quant.io.npz -O \
    run \
    --input-from-bundle $MODELS/inceptionv1_quant.io.npz \
    --allow-random-input \
    --assert-output-bundle $MODELS/inceptionv1_quant.io.npz

echo $WHITE     audio $NC

$CACHE_FILE ARM-ML-KWS-CNN-M.pb
$TRACT_RUN $MODELS/ARM-ML-KWS-CNN-M.pb \
    -O -i 49,10,f32 --partial \
    --input-node Mfcc \
    run -q \
    --allow-random-input

$CACHE_FILE $MODELS/GRU128KeywordSpotter-v2-10epochs.onnx
$TRACT_RUN $MODELS/GRU128KeywordSpotter-v2-10epochs.onnx \
    -O run -q \
    --allow-random-input \
    --assert-output-fact 1,3,f32

$CACHE_FILE en_libri_real/model.onnx
$TRACT_RUN $MODELS/en_libri_real/model.onnx \
    --output-node output \
    --edge-left-context 5 --edge-right-context 15 \
    --input-facts-from-bundle $MODELS/en_libri_real/io.npz \
    -O  \
    run \
    --input-from-bundle $MODELS/en_libri_real/io.npz \
    --approx approximate \
    --allow-random-input \
    --assert-output-bundle $MODELS/en_libri_real/io.npz
    
$CACHE_FILE mdl-en-2019-Q3-librispeech.onnx
$TRACT_RUN $MODELS/mdl-en-2019-Q3-librispeech.onnx \
    -O -i S,40,f32 --output-node output --pulse 24 \
    run -q \
    --allow-random-input
    
$CACHE_FILE hey_snips_v4_model17.pb
$TRACT_RUN $MODELS/hey_snips_v4_model17.pb \
    -i S,20,f32 --pulse 8 dump --cost -q \
    --assert-cost "FMA(F32)=2060448,Div(F32)=24576,Buffer(F32)=2920,Params(F32)=222250"

$TRACT_RUN $MODELS/hey_snips_v4_model17.pb -i S,20,f32 \
    dump -q \
    --assert-op-count AddAxis 0

$CACHE_FILE trunet_dummy.nnef.tgz
$TRACT_RUN --nnef-tract-core $MODELS/trunet_dummy.nnef.tgz dump -q

echo $WHITE     LLM $NC

TEMP_ELM=$(mktemp -d)
$CACHE_FILE 2024_06_25_elm_micro_export_with_kv_cache.nnef.tgz
$TRACT_RUN $MODELS/2024_06_25_elm_micro_export_with_kv_cache.nnef.tgz \
    --nnef-tract-core \
    --assert "S>0" --assert "P>0" --assert "S+P<2048" \
    dump -q --nnef $TEMP_ELM/with-asserts.nnef.tgz
$TRACT_RUN --nnef-tract-core $TEMP_ELM/with-asserts.nnef.tgz dump -q
rm -rf $TEMP_ELM

for t in harness/pre-optimized-graphes/*
do
    ( cd $t ; ./runme.sh)
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

