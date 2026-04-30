#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --set T=6 . run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — future-window banded mask, output stream.delay = 2 tokens
# (= L*P with L=1, P=2).  The CLI assertion path uses `pulse.delay`
# (now correctly rescaled through the boundary merge reshape) to skip
# the warmup tokens before comparing against the batch reference.
$TRACT_RUN --nnef-tract-core . --pulse 'T=2' run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
