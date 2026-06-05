#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch run and streaming compare: smoke test that pulsified model
# loads and produces the same output as the batched one.
$TRACT_RUN --nnef-tract-core . \
    -t 'set_symbols(values: {"S": 8})' \
    run --allow-random-input -q

$TRACT_RUN --nnef-tract-core . --pulse 4 compare \
    --stream --allow-random-input -q

# Body / outer fact consistency: the Pulsifier substitutes the stream
# symbol S in the outer wire facts; it must do the same on the Scan
# body source facts. Without the fix in pulse/src/ops/scan.rs the Full
# slot's body source keeps `S,4,F32` while the outer wire is
# `4,4,F32 [pulse axis:0 ...]`; assert here that the body source is
# a concrete shape (no residual stream symbol).
DUMP=$($TRACT_RUN --nnef-tract-core . --pulse 4 --pass pulse dump 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
if echo "$DUMP" | grep -A 1 '\[loop\].*Source.*full_y' | grep -qE '\bS\b'; then
    echo "ERROR: Scan body source for 'full_y' still carries the stream symbol after pulse pass."
    echo "       Pulsifier did not substitute it in the Scan body."
    echo "$DUMP" | grep -A 1 '\[loop\].*Source.*full_y'
    exit 1
fi
