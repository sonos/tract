#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for pulse in 2 4 6
do
    $TRACT_RUN --nnef-tract-core . --pulse $pulse compare \
        --stream --allow-random-input -q
done
