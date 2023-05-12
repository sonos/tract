#!/bin/sh

if [ -e /proc/cpuinfo ]
then
    grep "^flags" /proc/cpuinfo | head -1 | \
        grep --color=always '\(s\?sse[0-9_]*\|fma\|avx512[^ ]*\)'
fi

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

for c in data linalg core nnef hir onnx pulse onnx-opl pulse-opl
do
    cargo -q test $CARGO_EXTRA -q -p tract-$c
done
# doc test are not finding libtensorflow.so
if ! cargo -q test $CARGO_EXTRA -q -p tract-tensorflow --lib $ALL_FEATURES
then
    # this crate triggers an incremental bug on nightly.
    cargo clean -p tract-tensorflow
    cargo -q test $CARGO_EXTRA -q -p tract-tensorflow --lib $ALL_FEATURES
fi

if [ -n "$SHORT" ]
then
    exit 0
fi

OLD_CACHEDIR=$CACHEDIR
mkdir -p $CACHEDIR/big
export CACHEDIR=$CACHEDIR/big

cargo -q test $CARGO_EXTRA -q -p tract-rs

cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p core-proptest-pulse $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p lstm-proptest-onnx-vs-tf $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p nnef-inceptionv3 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-inceptionv3 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-mobilenet-v2 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-moz-deepspeech $ALL_FEATURES
if [ -n "$GITHUB_ACTIONS" ]
then
    rm -r $OLD_CACHEDIR/big
fi
CACHEDIR=$OLD_CACHEDIR
