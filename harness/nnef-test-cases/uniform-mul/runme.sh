#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$TRACT_RUN . dump -q --nnef-graph found

version=`cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] '`
perl -pi -e "s/$version/0.18.3-pre/" found

diff -u expected found

# Check result is as expected
$TRACT_RUN --nnef-tract-core . run --input-from-bundle io.npz --steps --assert-output-bundle io.npz
