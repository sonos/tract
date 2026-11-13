#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
for stage in load declutter optimize
do
    $TRACT_RUN --nnef-tract-core . --pass "$stage" \
        -i x:4,f32=1.55,-1.55,12.75,-12.75 run \
        --assert-output result:4,i32=15,-15,128,-128 --approx exact
done
