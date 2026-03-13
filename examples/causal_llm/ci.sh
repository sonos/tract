#!/bin/bash

set -e
set -o pipefail
set -x

wget https://s3.amazonaws.com/tract-ci-builds/tests/llm/541/Qwen--Qwen3-1.7B-q40ef16/Qwen--Qwen3-1.7B-q40ef16.nnef.tgz
wget https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json

OUTPUT=$(cargo run -p causal_llm --bin complete --profile opt-no-lto -- \
    -t "tokenizer.json" -m Qwen--Qwen3-1.7B-q40ef16.nnef.tgz -n 20 "The capital of France is")

echo "Output: $OUTPUT"

if [ -z "$OUTPUT" ]
then
    echo "ERROR: empty output"
    exit 1
fi
