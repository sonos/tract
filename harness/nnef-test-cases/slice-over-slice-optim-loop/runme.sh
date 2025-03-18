#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

# no timeout during recompilation
$TRACT_RUN --version

timeout 3 $TRACT_RUN --nnef-tract-core .
