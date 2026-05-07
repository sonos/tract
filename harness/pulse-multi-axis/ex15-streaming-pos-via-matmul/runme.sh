#!/bin/sh

# ex15-streaming-pos-via-matmul — known-failing repro of the
# blockify rel-pos chunked rewrite when the pos source is reached
# through a multi-input op (matmul) instead of a direct external.
# Batch passes; pulse compiles but `tract run` panics on input shape
# computation because `PulsedAxisSlice.skip` got the wrapped value
# `(−2 as i64) as usize = 18446744073709551614`.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=4 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
