#!/bin/sh

# ex10: inverted Iff convention — condition=True means masked out (fill=-inf).
#
# select(~window_mask, -inf, scores)  vs ex09's  select(window_mask, scores, -inf)
#
# This exposes two gaps that must be fixed before the encoder pulsifies:
#
# Gap 1 — PropagateRoi only annotates inputs[1] (true-branch = scores in the
#   standard convention).  Here scores are at inputs[2] (false-branch), so
#   PropagateRoi must detect the inverted convention and annotate inputs[2].
#
# Gap 2 — not(window_mask) produces a UniformTDim expression `1 + -1*cw` that
#   the UniformTDim pulsifier does not recognise.  It must handle negated
#   chunk-window expressions and emit an all-False mask of shape [P, KW].
#
# Expected results:
#   batch run   → PASS  (numpy reference matches graph evaluation)
#   compare --stream → FAIL until both gaps above are fixed

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
