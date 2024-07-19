#!/bin/sh

WHITE='\033[1;37m'
NC='\033[0m' # No Color

if [ -e /proc/cpuinfo ]
then
    grep "^flags" /proc/cpuinfo | head -1 | \
        grep --color=always '\(s\?sse[0-9_]*\|fma\|f16c\|avx[^ ]*\)'
fi

set -x

ROOT=$(dirname $0)/..
. $ROOT/.travis/ci-system-setup.sh

set -e

if [ `arch` = "x86_64" -a "$RUST_VERSION" = "stable" ]
then
    ALL_FEATURES=--all-features
fi

set +x

cargo -q test $CARGO_EXTRA -q -p tract-rs
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p core-proptest-pulse $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p lstm-proptest-onnx-vs-tf $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p nnef-inceptionv3 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-inceptionv3 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-mobilenet-v2 $ALL_FEATURES
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-moz-deepspeech $ALL_FEATURES
