#!/bin/sh

cd `dirname $0`
set -x

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$TRACT_RUN . --concretize-stream-dim 12 --nnef-tract-pulse dump
