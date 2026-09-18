#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN model.onnx --set batch_size=1 run --allow-random-input \
    --assert-output-fact 1,1,f32

$TRACT_RUN --onnx-ignore-output-shapes --set 'batch_size=2*batch_size' model.onnx \
    run --set batch_size=3 --allow-random-input --assert-output-fact 6,1,f32
