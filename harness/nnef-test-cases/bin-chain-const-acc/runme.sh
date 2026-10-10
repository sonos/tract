#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

trap 'rm -f ref.npz' EXIT

$TRACT_RUN . run --allow-random-input --save-outputs ref.npz
$TRACT_RUN . -O run --allow-random-input --assert-output-bundle ref.npz
