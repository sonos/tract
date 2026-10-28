#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# The squeeze after the tanh is hoisted above it and absorbed by the matmul store.
$TRACT_RUN . -O dump --assert-op-count IntoShape 0
