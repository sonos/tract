#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . run --input-from-bundle io.npz --assert-output-bundle io.npz --approx approximate

for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda|--metal|--runtime=wgpu) ;; *) continue;; esac
    $TRACT_RUN . $rt run --input-from-bundle io.npz --assert-output-bundle io.npz --approx approximate
done
