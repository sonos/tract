#!/bin/sh

set -ex

ROOT=$(dirname $(realpath $0))/..
. $ROOT/.travis/ci-system-setup.sh

opset=onnx_"${1:-1_13_0}"

cargo -q test -p test-unit-core $CARGO_EXTRA -q 
cargo -q test -p test-onnx-core $CARGO_EXTRA -q --no-default-features --features $opset
cargo -q test -p test-nnef-cycle $CARGO_EXTRA -q --no-default-features
