#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --set T=6 . run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — block-diagonal mask + EinSum terminator, no output delay.
$TRACT_RUN --nnef-tract-core . --pulse 'T=2' run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
