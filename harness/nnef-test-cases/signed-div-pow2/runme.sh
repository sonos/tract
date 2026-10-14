#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . -i 'input:2,i32=-3,3' run --assert-output 'output:2,i32=-1,1' --approx exact
$TRACT_RUN . -O -i 'input:2,i32=-3,3' run --assert-output 'output:2,i32=-1,1' --approx exact
