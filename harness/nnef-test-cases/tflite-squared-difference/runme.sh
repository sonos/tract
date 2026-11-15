#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# model.tflite holds SQUARED_DIFFERENCE on two inputs of different rank; io.npz is the output
# of the TFLite interpreter, see gen-model.py.
$TRACT_RUN model.tflite run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
