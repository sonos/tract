#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# io.npz: numpy's deconvolution of a 12-frame input, 3 pulses of 4.
for rt in "" $TRACT_RUNTIMES
do
    $TRACT_RUN --nnef-tract-core . $rt run --approx approximate \
        --input-from-bundle io.npz --assert-output-bundle io.npz
    $TRACT_RUN --nnef-tract-core . --pulse 4 $rt run --approx approximate \
        --input-from-bundle io.npz --assert-output-bundle io.npz
done
