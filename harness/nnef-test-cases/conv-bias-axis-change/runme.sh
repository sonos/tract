#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

inputs="--input a:2,3,3,f32=1,-2,3,0.5,4,-1,2,0,-3,1.5,-0.5,2,3,-4,1,0,2.5,-1.5 \
    --input bias_2d:1,2,f32=0.5,-1 --input bias_3d:2,1,f32=-0.25,2"

reference=`mktemp -d`
trap "rm -rf $reference" EXIT
$TRACT_RUN . $inputs --pass load run --save-outputs-npz $reference/io.npz
$TRACT_RUN . $inputs -O run --assert-output-bundle $reference/io.npz --approx approximate
