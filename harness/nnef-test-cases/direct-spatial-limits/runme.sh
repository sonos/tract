#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . -O dump --assert-op-count DirectSpatialConv 0
$TRACT_RUN . -O run --allow-random-input -q
