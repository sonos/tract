#!/bin/sh

# Batch + multi-head sliding-window attention.
# B=1, H=2, T=8, P=2 (chunk_size), left_chunks=1, Dh=4
#
# Input:  qkv [1, 2, S, 12]  streaming on axis 2
# Output: [1, 2, S, 4]
#
# Steps:
#   1. Generate reference inputs and batch output
#   2. Run the batch graph — sanity check
#   3. Pulsify and compare streaming output against batch

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
