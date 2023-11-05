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

cargo check -p tract-tflite
cargo -q test -p test-tflite $CARGO_EXTRA -q 
