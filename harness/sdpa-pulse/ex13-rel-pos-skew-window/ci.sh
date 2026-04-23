#!/bin/sh

# ex13: Transformer-XL relative-position attention with skew trick, left_chunks=1.
#
# Parameters: B=1, T=8, P=2, left_chunks=1, W=(left_chunks+1)*P=4, Dh=4, H=1
#
# The fix: PropagateRoi propagates the chunk-window ROI backward from the
# attention mask through the full pos_scores chain (Add → Slice → Reshape →
# DynSlice → Reshape → Pad → EinSum → R-gather).  Per-operator input_roi hooks
# on Slice, AxisOp, Pad, DynSlice and EinSum drive the propagation.  The Slice
# pulsifier then uses the ROI to extend each slice by L*P in the right direction:
#   - fixed-start slices (pos_sliced, pos_scores): extend end by L*P
#   - center-anchored slices (r from r_full, start=center-S): extend start back by L*P

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
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very
