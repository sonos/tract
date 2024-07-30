#!/bin/sh

set -x

cd $(dirname $0)

./test-published-crates.sh
./test-rt.sh
./test-harness.sh
