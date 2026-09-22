#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# tract.name outranks the graph id it is loaded from, and is written back as
# the graph id of the next serialization.
$TRACT_RUN . dump -q --nnef-graph found

version=`cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] '`
perl -pi -e "s/$version/0.18.3-pre/" found

diff -u expected found
