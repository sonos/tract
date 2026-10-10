#!/bin/sh

WHITE='\033[1;37m'
NC='\033[0m' # No Color

set -e

ROOT=$(realpath $(dirname $(realpath $0))/../..)
. $ROOT/.travis/ci-system-setup.sh

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

$CACHE_FILE mobilenet_v2_1.4_224_frozen.pb
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

$CACHE_FILE GRU128KeywordSpotter-v2-10epochs.onnx
$TRACT_RUN $MODELS/GRU128KeywordSpotter-v2-10epochs.onnx \
    -O run -q \
    --allow-random-input \
    --assert-output-fact 1,3,f32

$CACHE_FILE mdl-en-2019-Q3-librispeech.onnx mdl-en-2019-Q3-librispeech.io.npz
$TRACT_RUN $MODELS/mdl-en-2019-Q3-librispeech.onnx \
    --input-facts-from-bundle $MODELS/mdl-en-2019-Q3-librispeech.io.npz \
    -O --output-node output \
    run -q \
    --input-from-bundle $MODELS/mdl-en-2019-Q3-librispeech.io.npz \
    --assert-output-bundle $MODELS/mdl-en-2019-Q3-librispeech.io.npz \
    --approx approximate
$TRACT_RUN $MODELS/mdl-en-2019-Q3-librispeech.onnx \
    -O -i S,40,f32 --output-node output --pulse 24 \
    run -q \
    --input-from-bundle $MODELS/mdl-en-2019-Q3-librispeech.io.npz \
    --assert-output-bundle $MODELS/mdl-en-2019-Q3-librispeech.io.npz \
    --approx approximate
    
$CACHE_FILE hey_snips_v4_model17.pb
$TRACT_RUN $MODELS/hey_snips_v4_model17.pb \
    -i S,20,f32 --pulse 8 dump --cost -q \
    --assert-cost "FMA(F32)=2060448,Div(F32)=24576,Buffer(F32)=2920,Params(F32)=222251"

$TRACT_RUN $MODELS/hey_snips_v4_model17.pb -i S,20,f32 \
    dump -q \
    --assert-op-count AddAxis 0

$CACHE_FILE trunet_dummy.nnef.tgz
$TRACT_RUN --nnef-tract-core $MODELS/trunet_dummy.nnef.tgz dump -q
# --approx approximate: the GRU gate einsums (k=512 contraction) legitimately
# vary by ~1 ULP between the batched and pulsed paths with matmul reduction
# order; the default Close check is too tight.
$TRACT_RUN --nnef-tract-core $MODELS/trunet_dummy.nnef.tgz --pulse 1 \
    compare --stream --allow-random-input -q --approx approximate

echo $WHITE     LLM $NC

TEMP_ELM=$(mktemp -d)
$CACHE_FILE 2024_06_25_elm_micro_export_with_kv_cache.nnef.tgz
$TRACT_RUN $MODELS/2024_06_25_elm_micro_export_with_kv_cache.nnef.tgz \
    --nnef-tract-core \
    --assert "S>0" --assert "P>0" --assert "S+P<2048" \
    dump -q --nnef $TEMP_ELM/with-asserts.nnef.tgz
$TRACT_RUN --nnef-tract-core $TEMP_ELM/with-asserts.nnef.tgz dump -q
rm -rf $TEMP_ELM
