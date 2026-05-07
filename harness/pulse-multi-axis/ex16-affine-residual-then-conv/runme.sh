#!/bin/sh

# ex16-affine-residual-then-conv — known-failing repro of the
# Broadcast-TDim-in-pulse-value issue: stride-conv pulsifier emits
# per-pulse `k` while Range pulsifier emits per-pulse `1+k` for the
# same affine streaming dim, and the downstream conv pulsifier
# rejects the reconciled Broadcast pulse value.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

$TRACT_RUN --nnef-tract-core --set T=4 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

$TRACT_RUN --nnef-tract-core . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
