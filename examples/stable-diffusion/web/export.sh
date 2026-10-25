#!/bin/sh
# The graph tars the demo loads, from a Hugging Face snapshot of
# stable-diffusion-v1-5/stable-diffusion-v1-5 (the revision site/manifest.json pins):
#
#   ./export.sh <hf snapshot dir> <out dir>
#
# export.py writes weight-free ONNX graphs into the snapshot; tract then dumps each as an NNEF
# tar whose weights are tract_core_named_weight references to the safetensors tensors.

set -e
HF=`realpath $1`
OUT=`realpath -m $2`
cd `dirname $0`
: ${TRACT_RUN:=cargo run -q --release -p tract-cli --}

python export.py $HF
mkdir -p $OUT
for c in text_encoder:text_encoder unet:unet vae:vae_decoder
do
    dir=${c%%:*}
    name=${c##*:}
    $TRACT_RUN $HF/$dir/$name.onnx --onnx-lazy-external-data dump -q --nnef-tar $OUT/$name.nnef.tar
done
ls -l $OUT
