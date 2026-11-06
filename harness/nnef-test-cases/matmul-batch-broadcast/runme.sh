#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for rt in "" $TRACT_RUNTIMES
do
    $TRACT_RUN --nnef-tract-core . $rt run --input-from-bundle io.npz \
        --assert-output-bundle io.npz --approx very -q
done
