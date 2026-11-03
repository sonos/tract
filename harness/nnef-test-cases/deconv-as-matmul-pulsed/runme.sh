#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for rt in "" $TRACT_RUNTIMES
do
    $TRACT_RUN --nnef-tract-core . --pulse 2 $rt compare \
        --stream --allow-random-input --approx approximate -q
done

for rt in $TRACT_RUNTIMES
do
    [ "$rt" = "--cuda" ] || continue
    $TRACT_RUN --nnef-tract-core . --pulse 2 $rt dump -q --assert-op-count CudaDeconv 0
done
