#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --set T=6 . run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — banded mask + EinSum terminator (ex02 + ex03 mask).
# Terminator contracts axis_b (j); Blockify windows the j-side input
# (b) at the initiator AND the j-side auxiliary (c) at the terminator,
# both with `start = -upper` (past+current relative to kept axis i).
# Output stream.delay = 0 (causal in i).
$TRACT_RUN --nnef-tract-core . --pulse 'T=2' run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
