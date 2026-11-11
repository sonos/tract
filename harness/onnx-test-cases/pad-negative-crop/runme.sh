#!/bin/sh
set -eu
cd "$(dirname "$0")"
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
for optimize in '' '-O'
do
    $TRACT_RUN constant.onnx $optimize \
        -i 'x:4,f32=1,2,3,4' run \
        --assert-output 'y:4,f32=2,3,4,0'
    $TRACT_RUN reflect.onnx $optimize \
        -i 'x:4,f32=1,2,3,4' run \
        --assert-output 'y:5,f32=2,3,4,3,2'
    $TRACT_RUN axes.onnx $optimize \
        -i 'x:2,3,f32=1,2,3,4,5,6' run \
        --assert-output 'y:2,3,f32=2,3,0,5,6,0'
done
