#!/bin/sh
# The graph tars the demo loads, from torch_to_nnef's export of
# nvidia/nemotron-3.5-asr-streaming-0.6b (as ../ci.sh makes it, in assets/model) and the Hugging
# Face model.safetensors at the revision site/manifest.json pins:
#
#   ./export.sh <t2n model dir> <model.safetensors> <out dir>
#
# export.py rewrites each archive so its weights are tract_core_named_weight references to the
# safetensors tensors; about 50 MB the safetensors file does not hold stay in the encoder's tar.

set -e
T2N=`realpath $1`
ST=`realpath $2`
OUT=`realpath -m $3`
cd `dirname $0`
python3 export.py $ST $OUT $T2N/preprocessor.nnef.tgz $T2N/encoder.nnef.tgz $T2N/decoder.nnef.tgz $T2N/joint.nnef.tgz
ls -l $OUT
