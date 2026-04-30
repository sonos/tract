#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# u8 input → i8 output, both with the same scale, zero-point shifted by 128.
# Cast must subtract 128 from each byte to land the same real-valued range
# in the signed representation.
$TRACT_RUN --nnef-tract-core . dump --assert-op-count Cast 1
$TRACT_RUN --nnef-tract-core . run --allow-random-input
