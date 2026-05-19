#!/bin/sh

# ex13-padmask-broadcast-transpose — minimal repro of the pad-mask
# outer-AND pattern that encoder exports build for the 2D validity mask.
#
# Source layout: a 1D per-frame validity mask `pad [T]` is unsqueezed,
# broadcast to `[T, T]`, transposed, then AND'd against the original
# broadcast — yielding `pad_2d[i, j] = pad[i] AND pad[j]`.  Combined with
# a banded-causal block mask (P=2, L=1 → W=4), the AND lifts a 1D
# streaming source into a multi-T-axis score-shape wire, making it the
# section's initiator.
#
# Two passes handle this end-to-end:
# 1. `core/array/broadcast` declutter swaps each `Broadcast → AxisOp`
#    branch through under fan-out, then the existing single-successor
#    `Broadcast → TypedBinOp` elimination subsumes both broadcasts —
#    the AND ends up consuming `[1, T]` and `[T, 1]` directly.
# 2. Blockify's generic chunked-`TypedBinOp` section-initiator handler
#    chunk-splits each AND input on its streaming axis, windows the one
#    whose axis lands on the contracted (K) side, and wires the chunked
#    AND with implicit broadcasting at the chunked rank.
#
# Batch and pulsified both pass.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=4 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
