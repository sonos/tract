#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for rt in "" $TRACT_RUNTIMES
do
    $TRACT_RUN --nnef-tract-core . --pulse 4 $rt compare \
        --stream --allow-random-input -q
done
