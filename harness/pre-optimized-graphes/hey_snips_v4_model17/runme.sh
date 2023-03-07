#!/bin/sh

cd `dirname $0`

set -ex

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=../../../.cached
fi

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$TRACT_RUN $CACHEDIR/hey_snips_v4_model17.pb -i S,20,f32 --pulse 8 --nnef-tract-pulse --nnef-extended-identifier dump -q --nnef-graph found

version=`cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] '`
perl -pi -e "s/$version/0.19.3-pre/" found

diff -u expected found
