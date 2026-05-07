#!/bin/sh

# Block attention test: L = P = 2, T = 6
# chunked_limited, chunk=2, left_chunks=0 (bidirectional within chunk only)
# No relative position encoding.
#
# Steps:
#   1. Generate the block-diagonal mask and random Q/K/V inputs (via gen-inputs.py)
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
# Note: --input-from-bundle is intentionally omitted here.  handle_stream
# generates a fixed random input (seed 21242) and compare() must use the same
# random input so the two sides are consistent.  Intermediate nodes (scores,
# attn) may have their singleton streaming axis removed by ChangeAxes, so we
# only compare nodes where the streaming axis is still present.
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    --pulse 1 \
    compare \
    --stream \
    --allow-random-input \
    --approx very
