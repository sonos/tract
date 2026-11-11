#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . -t f32_to_f16 -O dump --assert-output-fact "2,F16"
$TRACT_RUN . -t 'f32_to_f16(keep_io: true)' -O dump --assert-output-fact "2,F32"
$TRACT_RUN . -t 'f32_to_f16(keep_io: true)' -O run --allow-random-input -q
