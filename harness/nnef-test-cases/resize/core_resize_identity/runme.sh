#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Before declutter the resize runs, and must leave the input unchanged.
$TRACT_RUN --nnef-tract-core . --pass load run --allow-random-input \
    --assert-op-count Resize 1 \
    --assert-output 'mismatch:1,1,1,1,f32=0'

# Declutter removes it.
$TRACT_RUN --nnef-tract-core . -O run --allow-random-input \
    --assert-op-count Resize 0 \
    --assert-output 'mismatch:1,1,1,1,f32=0'
