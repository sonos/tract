#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

rm -rf found
$TRACT_RUN --nnef-tract-core ./model dump -q --nnef-dir found

version=$(cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] ')
perl -pi -e "s/$version/0.21.2-pre/" found/graph.nnef

diff expected found
