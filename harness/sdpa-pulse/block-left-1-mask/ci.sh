#!/bin/sh

# Flat-token sliding-window attention with computed chunk mask.
# T=8 tokens, P=2 (chunk size), left_chunks=1, Dh=4
#
# Mask is computed entirely inside the graph from range/cast/div/floor/sub/le/ge/and,
# mirroring the Nemotron encoder mask construction.  The 'length' input of the real
# encoder is replaced by shape_of(qkv)[0].
#
# Steps:
#   1. Generate reference Q/K/V inputs and batch output
#   2. Run the batch graph — sanity check
#   3. Pulsify (pulse_size=P=2) and run streaming — assert output matches batch
#      NOTE: step 3 requires uniform_tdim to propagate through the mask chain so
#      that FoldUniformMask can fold the Iff nodes and expose the windowed K/V
#      structure to the pulsifier.  It may fail until that is implemented.

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
