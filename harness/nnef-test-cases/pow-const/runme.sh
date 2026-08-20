#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# pow against a uniform exponent declutters to the unary PowConst, so the op has
# to survive serialization too: without a serializer the round trip below fails
# where the first load succeeds.
$TRACT_RUN --nnef-tract-core . dump --assert-op-count PowConst 1

rm -rf roundtrip
$TRACT_RUN --nnef-tract-core . dump --nnef-dir roundtrip
$TRACT_RUN --nnef-tract-core roundtrip run --allow-random-input
rm -rf roundtrip
