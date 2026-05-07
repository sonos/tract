#!/bin/sh

# ex12: Transformer-XL relative-position attention with the skew trick.
#
# Purpose: verify that the "relative-shift" skew chain pulsifies correctly:
#
#     Q @ R^T [T, 2T-1]  →  Pad [T, 2T]  →  Reshape [2T, T]
#     →  Slice rows 1..T  →  Reshape [T, 2T-1]  →  Slice [:, :T]
#
# The positional encoding R is dynamically sliced from a fixed variable table,
# so its shape [2T-1, Dh] contains the streaming symbol — mirroring the
# encoder's posEnc_posEmb pattern.  This exercises PulsedSkewReshape (case 3).
#
# Parameters: B=1, T=8, P=2, left_chunks=0, Dh=4, H=1
# left_chunks=0 ensures key_window=P so pos_scores[P,P] matches content_scores[P,P].
# Expected:
#   batch run   → PASS
#   compare --stream → PASS (skew reshape correctly pulsified)

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
    -t 'pulse(symbol: Some("S"), pulse: "2")' \
    compare \
    --stream \
    --allow-random-input \
    --approx very
