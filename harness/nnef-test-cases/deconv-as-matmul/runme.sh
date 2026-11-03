#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for rt in "" $TRACT_RUNTIMES
do
    $TRACT_RUN --nnef-tract-core . $rt run --approx approximate \
        --input-from-bundle io.npz --assert-output-bundle io.npz
done

for rt in $TRACT_RUNTIMES
do
    [ "$rt" = "--cuda" ] || continue
    $TRACT_RUN --nnef-tract-core . $rt dump -q --assert-op-count CudaDeconv 0
done
