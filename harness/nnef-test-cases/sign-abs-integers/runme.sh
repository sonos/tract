#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# io.npz holds the input and the four expected outputs, computed with numpy.sign
# and numpy.abs rather than by tract.
#
# --assert-op-count pins which leg went through which path: exactly one Sign and one
# Abs node are left, which are the live legs, so the constant legs must have been
# folded. Without it, a pass that folded both legs, or one that folded neither, would
# still assert the same values and hide which code path produced them.
$TRACT_RUN . run --input-from-bundle io.npz --assert-output-bundle io.npz \
    --assert-op-count Sign 1 \
    --assert-op-count Abs 1

$TRACT_RUN . -O run --input-from-bundle io.npz --assert-output-bundle io.npz \
    --assert-op-count Sign 1 \
    --assert-op-count Abs 1
