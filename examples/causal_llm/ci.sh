#!/bin/bash

set -e
set -o pipefail
set -x

ROOT=$(dirname $(dirname $(dirname $(realpath $0))))
. $ROOT/.travis/ci-system-setup.sh

generation=541
model_id=Qwen--Qwen3-1.7B-q40ef16
nnef=llm/$generation/$model_id/$model_id.nnef.tgz

$CACHE_FILE $nnef

TOKENIZER=$MODELS/llm/Qwen--Qwen3-1.7B-tokenizer.json
if [ ! -e "$TOKENIZER" ]
then
    mkdir -p $(dirname "$TOKENIZER")
    wget -q https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json -O "$TOKENIZER.tmp"
    mv "$TOKENIZER.tmp" "$TOKENIZER"
fi

OUTPUT=$(cargo run -p causal_llm --bin complete --profile opt-no-lto -- \
    -t "$TOKENIZER" -m "$MODELS/$nnef" -n 20 "The capital of France is")

echo "Output: $OUTPUT"

if [ -z "$OUTPUT" ]
then
    echo "ERROR: empty output"
    exit 1
fi
