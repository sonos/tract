#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Before declutter the gathers run as Gather, against the slice reference.
$TRACT_RUN . --pass load run --allow-random-input \
    --assert-op-count Gather 2 \
    --assert-output 'mismatch:1,1,f32=0'

# Declutter turns them into a slice that drops the axis.
$TRACT_RUN . run --allow-random-input \
    --assert-op-count Gather 0 \
    --assert-output 'mismatch:1,1,f32=0'

$TRACT_RUN . -O run --allow-random-input \
    --assert-op-count Gather 0 \
    --assert-output 'mismatch:1,1,f32=0'
