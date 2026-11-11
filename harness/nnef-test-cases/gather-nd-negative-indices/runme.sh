#!/bin/sh
set -eu
cd "$(dirname "$0")"
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
for optimize in '' '-O'
do
    $TRACT_RUN --nnef-tract-core . $optimize \
        -i 'data:2,3,f32=1,2,3,4,5,6' run \
        --assert-output 'many:3,f32=6,1,2' \
        --assert-output 'single:1,f32=6' \
        --assert-output 'slice:1,3,f32=4,5,6'
done
