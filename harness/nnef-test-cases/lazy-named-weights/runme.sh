#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# The weights live in a weights.safetensors this case does not carry: loaded lazily, they are
# never read, and the dump refers to them by name.
$TRACT_RUN model.onnx --onnx-lazy-external-data --nnef-tract-core dump -q --nnef-graph found

version=`cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] '`
perl -pi -e "s/$version/0.18.3-pre/" found

diff -u expected found
