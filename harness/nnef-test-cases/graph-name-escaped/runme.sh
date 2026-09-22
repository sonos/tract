#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

version=`cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] '`

# A name NNEF spells only with the extended syntax: the graph id is mangled and
# tract.name keeps the exact one.
$TRACT_RUN . dump -q --nnef-graph found
perl -pi -e "s/$version/0.18.3-pre/" found
diff -u expected found

# With the extended syntax the graph id carries it, and the property goes.
$TRACT_RUN --nnef-extended-identifier . dump -q --nnef-graph found.extended
perl -pi -e "s/$version/0.18.3-pre/" found.extended
diff -u expected.extended found.extended
