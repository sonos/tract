#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch run with the streaming symbol concretized.
$TRACT_RUN --nnef-tract-core . \
    -t 'set_symbols(values: {"S": 8})' \
    run --allow-random-input -q

# Streaming compare. `select` has no dedicated pulse pulsifier, so it goes
# through the generic PulseWrappingOp path; its two value inputs (delays 1 and
# 2) sit below the condition's delay 3, so sync_inputs inserts a Delay on each.
# Before the fix both Delays were named `output.Delay` and pulsification failed
# at graph compaction with "duplicate name for node ... output.Delay". This
# step pulsifies and compares against the batch run, so it fails without the
# fix and passes with it.
$TRACT_RUN --nnef-tract-core . --pulse 4 compare \
    --stream --allow-random-input -q
