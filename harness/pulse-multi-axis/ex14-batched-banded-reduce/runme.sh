#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

PULSE='-t set_symbols(values:{"B":2}) -t pulse(symbol:Some("T"),pulse:"2")'

# Batch
$TRACT_RUN --nnef-tract-core --set T=6 --set B=2 . run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified. The mask chain windows into a Delay/PulsePad pair with no batch
# axis: correct here, since every stream of a batched run is at the same
# position, but it is a buffer they share.
$TRACT_RUN --nnef-tract-core . $PULSE run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN --nnef-tract-core . $PULSE dump -q \
    --assert-op-count Delay 4 --assert-op-count PulsePad 2

# Batchified first: the mask chain carries the batch axis, so the pair it
# windows into is addressable per stream. Same numbers, and the same state
# count — a batch axis that perturbs the chunk frame shows up here as extra
# Delay states, not as a wrong result.
$TRACT_RUN --nnef-tract-core . -t 'batchify_data_free(symbol:Some("B"))' $PULSE \
    run --approx approximate --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN --nnef-tract-core . -t 'batchify_data_free(symbol:Some("B"))' $PULSE dump -q \
    --assert-op-count Delay 4 --assert-op-count PulsePad 2
