#!/bin/sh

# Block attention test: left_chunks=1, P=2, C=4
# chunked_limited, chunk=2, left_chunks=1 (bidirectional within 2-chunk window)
# No relative position encoding.
#
# Each chunk c attends to its own P tokens and the P tokens of chunk c-1.
# The previous-chunk K/V is zero for c=0 (delay initialised to zero).
#
# Steps:
#   1. Generate reference Q/K/V inputs and batch output (via gen-inputs.py)
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
