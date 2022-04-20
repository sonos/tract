#!/bin/sh

cd `dirname $0`
set -x

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$TRACT_RUN . dump -q --nnef-graph found

diff expected found