#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

# Check result is as expected
# bug appear only if model optimized
timeout 10 $TRACT_RUN --nnef-tract-core .
