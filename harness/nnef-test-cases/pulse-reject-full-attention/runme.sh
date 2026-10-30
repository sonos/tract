#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core . --set T=6 run --allow-random-input -q

if $TRACT_RUN --nnef-tract-core . --pulse T=2 dump -q 2> stderr.txt
then
    echo "pulsified unmasked attention"
    exit 1
fi
grep -q "quadratic in streaming symbol" stderr.txt
rm stderr.txt
