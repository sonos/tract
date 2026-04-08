#!/bin/sh

# Block attention with boolean mask: L = P = 2, T = 8
# Block-diagonal, all-true mask — exercises Iff + softmax in streaming.
# No relative position encoding.
#
# Steps:
#   1. Generate reference Q/K/V inputs, all-true mask, and batch output
#   2. Run the batch graph — sanity check
#   3. Pulsify and run streaming — assert output matches batch

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# ── 1. generate inputs ───────────────────────────────────────────────────────
python3 gen-inputs.py

# ── 2. batch run (reference) ─────────────────────────────────────────────────
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very

# ── 3. pulsed run — 1 chunk per pulse, compare against batch ─────────────────
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    --pulse 1 \
    compare \
    --stream \
    --allow-random-input \
    --approx very
