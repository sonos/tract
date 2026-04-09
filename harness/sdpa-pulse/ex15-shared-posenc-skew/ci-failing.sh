#!/bin/sh

# ex15: fixed symmetric RPE [15,4] with left_chunks=3, W=8 > 2P-1=3.
# Same structure as ex14-rel-pos-skew-large-table but with larger left_chunks.
# Reproduces the encoder skew trick failure: pos_scores = [P, P] vs content = [P, W].

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
    -t 'pulse: "4")' \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very
