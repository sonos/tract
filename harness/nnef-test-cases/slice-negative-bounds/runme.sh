#!/bin/sh
set -eu
cd "$(dirname "$0")"
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
for optimize in '' '-O'
do
    $TRACT_RUN --nnef-tract-core . $optimize \
        -i 'data:1,5,2,f32=0,1,2,3,4,5,6,7,8,9' run \
        --assert-output 'last_two:1,2,2,f32=6,7,8,9' \
        --assert-output 'last:1,1,2,f32=8,9' \
        --assert-output 'middle:1,2,2,f32=2,3,4,5'
done
