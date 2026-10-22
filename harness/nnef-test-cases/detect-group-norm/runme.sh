#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . run --input-from-npz io.npz --assert-output-bundle io.npz --approx approximate -q \
    --assert-op-count GroupNorm 1 --assert-op-count RmsNorm 0

$TRACT_RUN . -O run --input-from-npz io.npz --assert-output-bundle io.npz --approx approximate -q
