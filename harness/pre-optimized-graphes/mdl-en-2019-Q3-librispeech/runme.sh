#!/bin/sh

set -ex

cd `dirname $0`

ROOT=$(dirname $(realpath $0))/../../..
. $ROOT/.travis/ci-system-setup.sh

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$CACHE_FILE mdl-en-2019-Q3-librispeech.onnx
$TRACT_RUN $MODELS/mdl-en-2019-Q3-librispeech.onnx -i S,40 --output-node output --pulse 24 --nnef-tract-pulse --nnef-extended-identifier dump -q --nnef-graph found

version=`cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] '`
perl -pi -e "s/$version/0.19.3-pre/" found

diff -u expected found
