#!/bin/sh

# ex11: ex09 with ScaledMaskedSoftmax instead of select+softmax.
#
# Purpose: verify that TypedOp::input_roi on ScaledMaskedSoftmax drives
# PropagateRoi → pulsify_qk (K Delay) and pulsify_av (V Delay).
#
# Parameters: B=1, H=2, T=8, P=2, left_chunks=1, Dh=4
# Expected:
#   batch run   → PASS
#   compare --stream → PASS (K and V delays inserted via input_roi ROI)

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
