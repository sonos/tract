#!/bin/sh

set -ex

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../.cached
fi

# useful as debug_asserts will come into play
cargo test -p tract-core
cargo test -p test-onnx-core -p test-nnef-cycle -p test-unit-core
