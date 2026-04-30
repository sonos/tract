#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --set T=6 . run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — causal banded mask (-1 ≤ diff ≤ 0).  PulsePad zero-fills
# the leading past-window slot so the streamed output's stream.delay
# is 0 and the output matches batch 1:1 from index 0.
$TRACT_RUN --nnef-tract-core . --pulse 'T=2' run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
