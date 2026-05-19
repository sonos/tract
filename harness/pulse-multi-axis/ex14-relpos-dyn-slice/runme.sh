#!/bin/sh

# ex14-relpos-dyn-slice — Transformer-XL relative-position chain at minimal
# scale: `slice(pos_enc_pe, begin=5-T, end=4+T)` produces a streaming-width
# `[2T-1, D]` rel-pos table; an einsum against the query yields `[T, 2T-1]`;
# the skew trick (pad → reshape → slice → reshape → slice) folds (via
# `detect_diag_gather`) to a single DiagGather op whose input carries the
# streaming-T symbol on *both* the query-row axis and the rel-pos R-axis.
#
# This is the smallest case that exercises the pre-blockify rel-pos rewrite
# (`rewrite_streaming_relpos_chains` in `pulse/src/blockify.rs`): the rewrite
# replaces the streaming-width slice with a constant-width window of
# `(L+2)·k - 1` rows centred on rel-pos zero, replays the intermediate chain
# with the narrower wire, and re-anchors the folded DiagGather's `offset` so
# the per-element semantics are preserved.  After the rewrite the existing
# DiagGather initiator handles the section.
#
# Batch passes; pulsified passes too once the rewrite is in place.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch — passes.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=4 . \
    -t transformers_detect_all \
    run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — currently fails (needs the pre-blockify rel-pos rewrite).
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . \
    -t transformers_detect_all \
    --pulse 'T=2' \
    run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
