#!/bin/sh

set -ex

which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y
rustup update

PATH=$PATH:$HOME/.cargo/bin

if [ ${RUST_VERSION:=stable} != "stable" ]
then
    rustup toolchain add $RUST_VERSION
fi
export RUSTUP_TOOLCHAIN=$RUST_VERSION

rustc --version

if [ `uname` = "Darwin" ]
then
    sysctl -n machdep.cpu.brand_string
    brew install coreutils
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

if [ `arch` = "x86_64" -a "$RUST_VERSION" = "stable" ]
then
    ALL_FEATURES=--all-features
fi

cargo -q test $CARGO_EXTRA -q -p tract-core -p tract-hir -p tract-onnx -p tract-linalg
# doc test are not finding libtensorflow.so
cargo -q test $CARGO_EXTRA -q -p tract-tensorflow --lib $ALL_FEATURES

if [ -n "$SHORT" ]
then
    exit 0
fi

cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p core-proptest-pulse $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p lstm-proptest-onnx-vs-tf $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p nnef-inceptionv3 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-inceptionv3 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-mobilenet-v2 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-moz-deepspeech $ALL_FEATURES

