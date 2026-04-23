#!/bin/sh

# Flat-token sliding-window attention with computed chunk mask + ALiBi position bias.
# T=8 tokens, P=2 (chunk size), left_chunks=1, Dh=4, slope=0.125
#
# Extends ex04-block-left-1-mask: pos_bias[i,j] = -0.125*(i-j) is added to scores
# before masking.  In windowed form pos_bias[p,l] = -slope*(L*P+p-l), which is
# constant across chunks and collapses to a precomputed [P,(L+1)*P] tensor at pulse time.
#
# Steps:
#   1. Generate reference Q/K/V inputs and batch output (with pos_bias)
#   2. Run the batch graph — sanity check
#   3. Pulsify and compare streaming output against batch
#      NOTE: step 3 requires FoldWindowAttention to be extended to handle
#      Iff(mask, Add(einsum_scores, pos_bias), -inf) rather than
#      Iff(mask, einsum_scores, -inf).  It will fail until that is implemented.

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
