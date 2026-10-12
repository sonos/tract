#!/bin/sh

set -e
set -x

cd $(dirname $0)

./test-published-crates.sh
if [ -n "$CI" ]
then
    cargo clean
fi
./test-rt.sh
if [ -n "$CI" ]
then
    cargo clean
fi
