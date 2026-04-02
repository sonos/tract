#!/bin/sh

# Sliding-window attention with chunk-level relative-position bias.
# T=8, P=2, Dh=4, left_chunks=1, slope=-0.5
#
# Position bias: v_bias[i,j] = -0.5 * (floor(i/P) - floor(j/P))
#
# The binary pulsifier fires on the sub(floor_i, floor_j) wire whose
# uniform_tdim = Div(🎯0, 2) − Div(🎯1, 2), exercising integer-division
# inside a TDim coordinate expression (new vs ex05's linear i−j).
#
# Steps:
#   1. Generate reference inputs and batch output
#   2. Run the batch graph — sanity check
#   3. Pulsify (--pulse 2) and compare streaming output against batch

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

# ── 3. pulsed run — P=2 tokens per pulse (1 chunk), compare against batch ────
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    --pulse 2 \
    compare \
    --stream \
    --allow-random-input \
    --approx very
