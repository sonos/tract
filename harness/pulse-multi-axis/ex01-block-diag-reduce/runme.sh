#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch — T concrete, full reference.
$TRACT_RUN --nnef-tract-core --set T=6 . run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — Blockify recognises the block-diagonal mask, rewrites the
# section, and pulse fires on the chunk axis with pulse 2 tokens.  The
# block-diagonal mask has zero output delay, so the streamed output
# matches the batch reference 1:1.
$TRACT_RUN --nnef-tract-core . --pulse 'T=2' run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
