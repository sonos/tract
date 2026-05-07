#!/bin/sh

# ex13-padmask-broadcast-transpose — minimal repro of encoder.p1's
# blockify failure on the pad-mask construction.
#
# Pattern (encoder layout): a 1D per-frame validity mask `pad [T]` is
# unsqueezed, broadcast to [T, T], transposed, then AND'd against the
# original broadcast — yielding the 2D mask `[i, j] = pad[i] AND pad[j]`.
# Combined with a banded-causal block mask (P=2, L=1 → W=4), the body
# `pad_T = Move(1, 2)` swaps two within-chunk axes that came from
# differently-chunked sources: position 1 is the broadcast-from-1 slot
# (size k=2) and position 2 is the windowed-streaming slot (size W=4).
# AND of `[S, 2, 4]` and `[S, 4, 2]` shape-mismatches and tract reconciles
# it as `[S, (2)#(4), (4)#(2)]` (Broadcast TDims), which then propagates
# through softmax and the V matmul, ultimately failing pulsification with:
#
#     Trying to substitute a 2*S,4,F32
#     by S,((2)#((2)#(4)))#(2),4,F32
#     as output #0 of #31 "output" EinSum.
#
# Same root cause as encoder.p1's `padMaskForAttMask_1` body op, at the
# smallest possible scale.  Batch passes; pulse fails.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch — passes.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=4 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — currently fails.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
