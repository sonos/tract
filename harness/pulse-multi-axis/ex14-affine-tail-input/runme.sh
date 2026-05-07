#!/bin/sh

# ex14-affine-tail-input — known-failing repro of pulse-mode handling
# of affine-tail streaming inputs.  Batch passes; pulse compiles but
# produces incorrect output (per-pulse 3 ≠ k=2 because the typed-side
# affine_trim Slice doesn't trim the per-pulse buffer).

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
