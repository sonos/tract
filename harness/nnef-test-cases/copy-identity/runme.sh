#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# No graph.quant: copy is pure identity, no Cast node should appear.
$TRACT_RUN --nnef-tract-core . dump --assert-op-count Cast 0
$TRACT_RUN --nnef-tract-core . run --allow-random-input
