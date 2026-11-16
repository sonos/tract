#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . --nnef-tract-core dump -q

for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda|--metal|--runtime=wgpu) ;; *) continue;; esac
    $TRACT_RUN . --nnef-tract-core $rt dump -q --cost
done
