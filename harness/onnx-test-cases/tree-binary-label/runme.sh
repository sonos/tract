#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
for model in probability signed post_transform; do
    for optimize in '' '-O'; do
        $TRACT_RUN "$model.onnx" $optimize -i 'x:2,1,f32=-1,1' --output-node label run --assert-output 'label:2,i64=11,21'
    done
done
