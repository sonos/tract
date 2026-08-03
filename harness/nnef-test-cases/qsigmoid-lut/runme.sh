#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# dequant (u8 -> f32) -> sigmoid -> requant (f32 -> u8) is a pointwise chain on
# a byte tensor: optimization must bake it into a single 256-entry lookup table,
# leaving no Cast behind.
$TRACT_RUN --nnef-tract-core . -O run --allow-random-input \
    --assert-op-count Cast 0 --assert-op-count LookupTable 1
