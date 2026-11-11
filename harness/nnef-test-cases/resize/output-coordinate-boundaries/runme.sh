#!/bin/sh
set -eu
cd "$(dirname "$0")"
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
for optimize in '' '-O'
do
    $TRACT_RUN --nnef-tract-onnx . $optimize \
        -i 'data:1,1,1,5,f32=1,2,3,4,5' \
        -i 'image:1,1,2,2,f32=1,2,3,4' run \
        --assert-output 'aligned:1,1,1,3,f32=1,3,5' \
        --assert-output 'cropped:1,1,1,3,f32=2,3,4' \
        --assert-output 'center:1,1,1,1,f32=3' \
        --assert-output 'pixel:1,1,1,1,f32=1'
done
