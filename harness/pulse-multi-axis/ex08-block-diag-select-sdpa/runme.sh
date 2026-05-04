#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=6 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — Q·Kᵀ → select(block-diag mask, scores, -inf) → softmax →
# attn·V.  This exercises the `MultiBroadcastTo` initiator path in
# Blockify: declutter folds `scores * 0.0 + -inf` to a scalar-(-inf)
# broadcast to `[T, T]`, which lands as a non-data initiator inside
# the multi-T-axis section.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
