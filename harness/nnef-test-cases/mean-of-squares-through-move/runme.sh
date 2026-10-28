#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# A transpose between the square and the mean blocks the MeanOfSquares fusion,
# and touches the model interfaces on both sides, so only moving it below the
# reduce lets the fusion happen.
$TRACT_RUN . dump --assert-op-count Reduce\<MeanOfSquares\> 1
$TRACT_RUN . -O run --input-from-bundle io.npz --assert-output-bundle io.npz
