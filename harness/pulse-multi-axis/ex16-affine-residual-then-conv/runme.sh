#!/bin/sh

# ex16-affine-residual-then-conv — was the known-failing repro of
# the Broadcast-TDim-in-pulse-value issue (stride-conv pulsifier
# emits per-pulse `k` while Range pulsifier emits per-pulse `1+k`
# for the same affine streaming dim, breaking any downstream conv
# pulsifier that reads `input_fact.pulse()` as a concrete value).
# Range pulsifier is now affine-aware: for typed `c + k·S` it emits
# per-pulse `k·pulse` (= conv convention), so paths through Pad and
# Range agree on per-pulse and the downstream pulsifier sees a
# concrete pulse value.
#
# Pulse mode emits `T = k·S` elements (= 4 for T=4, k=2, S=2);
# the typed `1+T` view of the reference has 5 elements.  The pulse
# `--drop-partial-pulse` flag wouldn't help here because there's no
# partial input pulse — the truncation is at the *output* tail.
# We compare only the prefix that pulse mode actually emits.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

$TRACT_RUN --nnef-tract-core --set T=4 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

$TRACT_RUN --nnef-tract-core . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
