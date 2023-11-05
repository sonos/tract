#!/bin/sh

set -ex

which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y

PATH=$PATH:$HOME/.cargo/bin

: "${RUST_VERSION:=stable}"
rustup toolchain add $RUST_VERSION
rustup default $RUST_VERSION

rustc --version

if [ `uname` = "Darwin" ]
then
    brew install coreutils
fi
if [ -n "$GITHUB_ACTIONS" ]
then
    pip install numpy
fi

# if [ `uname` = "Linux" -a -z "$TRAVIS" ]
# then
#     apt-get update
#     apt-get -y upgrade
#     apt-get install -y unzip wget curl python awscli build-essential git pkg-config libssl-dev
#     cargo --version || ( curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y )
# fi


if [ -z "$CACHEDIR" ]
then
    CACHEDIR=$(realpath `dirname $0`/../.cached)
fi

export CACHEDIR

opset=onnx_"${1:-1_13_0}"

cargo -q test -p test-onnx-core $CARGO_EXTRA -q --no-default-features --features $opset
cargo -q test -p test-onnx-nnef-cycle $CARGO_EXTRA -q --no-default-features
cargo -q test -p test-unit-core $CARGO_EXTRA -q 
