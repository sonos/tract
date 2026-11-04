#!/bin/sh
# The stage files the demo loads, from a snapshot of prism-ml/bonsai-image-ternary-4B-unpacked:
#
#   ./export.sh <hf snapshot dir> <out dir>
#
# Needs a tract CLI built with the wgpu runtime's dependencies at $TRACT, and python with torch,
# diffusers, transformers, onnx, onnxscript and safetensors.

set -e
HF=`realpath $1`
OUT=`realpath $2`
cd `dirname $0`
: ${TRACT:=cargo run -q --release -p tract-cli --}
: ${SIZES:=256x256 384x384 512x512 384x512 512x384 768x768}
TMP=$OUT/onnx
mkdir -p $TMP

BASE=${SIZES%% *}
for size in $SIZES
do
    for stage in transformer vae_decoder
    do
        ASSETS=$TMP HF_SNAPSHOT=$HF SIZE=$size DIT_TEXT_LEN=64 EXPORT_COMPONENT=$stage python ../export.py
        $TRACT $TMP/$stage.onnx --nnef-tract-core --nnef-tract-transformers dump -q --nnef-tar $OUT/$stage-$size.nnef.tar
        [ $size = $BASE ] || python split_stages.py add $OUT $stage $BASE $size
    done
done
for stage in transformer vae_decoder
do
    python split_stages.py finish $OUT $stage $BASE
done

ASSETS=$TMP HF_SNAPSHOT=$HF EXPORT_COMPONENT=text_encoder python ../export.py
$TRACT $TMP/text_encoder.onnx --nnef-tract-core --nnef-tract-transformers dump -q --nnef-tar $OUT/text_encoder.nnef.tar
rm -r $TMP
cp $HF/tokenizer/tokenizer.json $OUT/
