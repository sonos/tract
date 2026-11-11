#!/bin/sh
set -eu
cd "$(dirname "$0")"
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
for optimize in '' '-O'
do
    $TRACT_RUN --nnef-tract-core . $optimize \
        -i 'data:1,2,3,f32=1,2,3,4,5,6' run \
        --assert-output 'batched:1,f32=6' \
        --assert-output 'unbatched:1,f32=6'
done
