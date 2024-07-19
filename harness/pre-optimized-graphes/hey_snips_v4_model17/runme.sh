#!/bin/sh

cd `dirname $0`

ROOT=$(dirname $(realpath $0))/../../..
. $ROOT/.travis/ci-system-setup.sh

set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$CACHE_FILE hey_snips_v4_model17.pb
$TRACT_RUN $MODELS/hey_snips_v4_model17.pb -i S,20,f32 --pulse 8 --nnef-tract-pulse --nnef-extended-identifier dump -q --nnef-graph found

version=`cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] '`
perl -pi -e "s/$version/0.19.3-pre/" found

diff -u expected found
