#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch mode: S=8 -> conv output T = 1 + 8/2 = 5 frames; the add is fine.
$TRACT_RUN --nnef-tract-core . \
    -t 'concretize_symbols(values: {"S": 8})' \
    run --allow-random-input -q

# Streaming compare: pulse=4 -> conv produces 2 frames/step.  The
# tract_core_broadcast (shape=[1,1,S/2+1]) must also produce 2 frames/step.
# The MultiBroadcastTo pulsifier removes the constant boundary term so that
# per-pulse size = substitute(S→P) - substitute(S→0) = (1+P/2) - 1 = P/2.
$TRACT_RUN --nnef-tract-core . --pulse 4 compare \
    --stream --allow-random-input -q
