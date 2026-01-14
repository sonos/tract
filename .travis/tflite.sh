#!/bin/sh

set -ex

ROOT=$(dirname $(dirname $(realpath $0)))
. $ROOT/.travis/ci-system-setup.sh

if [ `uname` = "Darwin" ]
then
    brew install coreutils
fi
if [ -n "$GITHUB_ACTIONS" ]
then
    pip install numpy
fi

cargo check -p tract-tflite
cargo -q test -p test-tflite $CARGO_EXTRA -q 
