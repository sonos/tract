#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# The model's own layout, batch axis inside.
$TRACT_RUN --nnef-tract-core --set T=5 --set B=2 . run \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Batchified: the interface carries the batch on axis 0 both ways, and the moves
# that put it there are graph edits declutter absorbs -- no transpose is left.
$TRACT_RUN --nnef-tract-core . -t 'batchify(symbol:Some("B"))' dump -q \
    --assert-output-fact "B,T,1,F32" --assert-op-count MoveAxis 0

$TRACT_RUN --nnef-tract-core . -t 'batchify(symbol:Some("B"))' --set T=5 --set B=2 run \
    --input-from-bundle io-batch-first.npz --assert-output-bundle io-batch-first.npz
