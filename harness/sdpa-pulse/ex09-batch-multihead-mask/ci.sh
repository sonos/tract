#!/bin/sh

# ex09: batch + head dimensions in T×T masked attention.
# Scores [B, H, S, S], mask [B, 1, S, S].  Verifies ROI propagation
# through both batch and head axes.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very

$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    --pulse 2 \
    compare \
    --stream \
    --allow-random-input \
    --approx very
