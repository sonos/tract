#!/bin/bash

set -ex

LOCAL_DIR=$(dirname $(dirname $(realpath $0)))
echo $LOCAL_DIR

tmp=$(echo ${TMPDIR:-${TEMP:-${TMP:-/tmp}}})
venv=$tmp/venv-for-tract-assets

if [ ! -d $venv ]
then
    virtualenv -p python3.11 $venv
    source $venv/bin/activate
    (cd $LOCAL_DIR/llm-python/ ; pip install -e ".[cli]")
else
    source $venv/bin/activate
fi

MODELS=
MODELS="$MODELS apple/OpenELM-270M"
MODELS="$MODELS apple/OpenELM-1_1B"
MODELS="$MODELS TinyLlama/TinyLlama_v1.1"
MODELS="$MODELS microsoft/phi-1_5"
MODELS="$MODELS meta-llama/Llama-3.2-3B"

VARIANTS="f32f32 q40f32 q40ef32 f16f16 q40f16 q40ef16 "

for hf_id in $MODELS
do
    local_id=$(echo $hf_id | sed "s/\//--/g")

    for q in $VARIANTS
    do
         model=$local_id-$q
         rm -rf $model
         case $q in
             f32f32) EXPORT_ARGS= ;;
             f16f16) EXPORT_ARGS=-f16 ;;
             q40f32) EXPORT_ARGS="-c min_max_q4_0" ;;
             q40ef32) EXPORT_ARGS="-c min_max_q4_0_with_embeddings" ;;
             q40f16) EXPORT_ARGS="-c min_max_q4_0 -f16" ;;
             q40ef16) EXPORT_ARGS="-c min_max_q4_0_with_embeddings -f16" ;;
         esac

         python $LOCAL_DIR/llm-python/llm/cli/export_llm_from_torch_to_nnef.py \
             --sample-generation-total-size 100 \
             --no-verify \
             -s $hf_id -e $model $EXPORT_ARGS

          aws s3 cp $model/model/model.nnef.tgz s3://tract-ci-builds/tests/llm/current/$model/$model.nnef.tgz
          aws s3 cp $model/tests/prompt_io.npz s3://tract-ci-builds/tests/llm/current/$model/$model.p0s100.io.npz
          aws s3 cp $model/tests/text_generation_io.npz s3://tract-ci-builds/tests/llm/current/$model/$model.p99s1.io.npz
          if [ -e $model/tests/prompt_with_past_io.npz ]
          then
            aws s3 cp $model/tests/prompt_with_past_io.npz s3://tract-ci-builds/tests/llm/current/$model/$model.p50s50.io.npz
          fi
    done

done
