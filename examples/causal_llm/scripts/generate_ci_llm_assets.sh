#!/bin/bash

set -ex

tmp=$(echo ${TMPDIR:-${TEMP:-${TMP:-/tmp}}})
venv=$tmp/venv-for-tract-assets

if [ ! -d $venv ]
then
    virtualenv -p python3.13 $venv
    source $venv/bin/activate
    pip install 'torch-to-nnef[llm-tract]>=0.20.0' 'transformers>=4.51,<4.52'
else
    source $venv/bin/activate
fi

MODELS=
MODELS="$MODELS apple/OpenELM-270M"
MODELS="$MODELS meta-llama/Llama-3.2-1B-Instruct"
MODELS="$MODELS meta-llama/Llama-3.2-3B-Instruct"
MODELS="$MODELS meta-llama/Llama-3.1-8B-Instruct"
MODELS="$MODELS Qwen/Qwen2.5-7B-Instruct"
MODELS="$MODELS Qwen/Qwen3-1.7B"
MODELS="$MODELS Qwen/Qwen3-8B"

MODELS_F32_ALLOWED="meta-llama/Llama-3.2-1B-Instruct"

VARIANTS="f32f32 f16f16 q40ef16 "

for hf_id in $MODELS
do
    local_id=$(echo $hf_id | sed "s/\//--/g")

    for q in $VARIANTS
    do
	 if [ "$q" = "f32f32" ] && ! echo "$MODELS_F32_ALLOWED" | grep -q -w "$hf_id"; then
         echo "INFO: Skipping f32f32 for model $hf_id"
            continue
         fi
         model=$local_id-$q
         rm -rf $model
         case $q in
             f32f32) EXPORT_ARGS= ;;
             f16f16) EXPORT_ARGS="-dt f16" ;;
             q40f32) EXPORT_ARGS="-c min_max_q4_0" ;;
             q40ef32) EXPORT_ARGS="-c min_max_q4_0_with_embeddings" ;;
             q40ef16) EXPORT_ARGS="-c min_max_q4_0_with_embeddings -dt f16" ;;
         esac

         t2n_export_llm_to_tract \
             --sample-generation-total-size 100 \
             --no-verify \
             -s $hf_id -e $model $EXPORT_ARGS \
	     --reify-sdpa-operator

          aws s3 cp $model/model/model.nnef.tgz s3://tract-ci-builds/tests/llm/current/$model/$model.nnef.tgz
          aws s3 cp $model/tests/prompt_io.npz s3://tract-ci-builds/tests/llm/current/$model/$model.p0s100.io.npz
          aws s3 cp $model/tests/text_generation_io.npz s3://tract-ci-builds/tests/llm/current/$model/$model.p99s1.io.npz
          if [ -e $model/tests/prompt_with_past_io.npz ]
          then
            aws s3 cp $model/tests/prompt_with_past_io.npz s3://tract-ci-builds/tests/llm/current/$model/$model.p50s50.io.npz
          fi
    done

done
