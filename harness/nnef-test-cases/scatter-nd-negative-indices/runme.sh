#!/bin/sh
set -eu
cd "$(dirname "$0")"
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
for optimize in '' '-O'
do
    $TRACT_RUN --nnef-tract-core . $optimize \
        -i 'data:2,3,f32=1,2,3,4,5,6' run \
        --assert-output 'assigned:2,3,f32=20,2,3,4,5,10' \
        --assert-output 'added:2,3,f32=21,2,3,4,5,16' \
        --assert-output 'multiplied:2,3,f32=20,2,3,4,5,60' \
        --assert-output 'minimum:2,3,f32=1,2,3,4,5,6' \
        --assert-output 'maximum:2,3,f32=20,2,3,4,5,10'
done
