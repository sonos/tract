#!/bin/sh

cd `dirname $0`
set -x

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

rm -rf found
$TRACT_RUN model dump -q --nnef-dir found

diff expected found