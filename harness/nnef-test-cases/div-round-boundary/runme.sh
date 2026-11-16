#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
x=1.55,-1.55,12.75,-12.75
result=15,-15,128,-128
wide_x=$x
wide_result=$result
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    wide_x=$wide_x,$x
    wide_result=$wide_result,$result
done
for stage in load declutter optimize
do
    $TRACT_RUN --nnef-tract-core . --pass "$stage" \
        -i x:4,f32=$x run \
        --assert-output result:4,i32=$result --approx exact
    $TRACT_RUN --nnef-tract-core . --pass "$stage" \
        -i x:64,f32=$wide_x run \
        --assert-output result:64,i32=$wide_result --approx exact
done
