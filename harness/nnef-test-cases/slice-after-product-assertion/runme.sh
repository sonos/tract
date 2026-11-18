#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# 10 prefix rows sliced off 10 + 4 * MH * MW rows, with an assertion on MH * MW
$TRACT_RUN --nnef-tract-core . run --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN --nnef-tract-core . -O run --input-from-bundle io.npz --assert-output-bundle io.npz
