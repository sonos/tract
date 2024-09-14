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

for c in data linalg core nnef hir onnx pulse onnx-opl pulse-opl rs
do
    echo
    echo "$WHITE ### $c ### $NC"
    echo
    cargo -q test $CARGO_EXTRA -q -p tract-$c
done

if [ `uname` = "Darwin" -a -z "$CI" ]
then
    echo
    echo "$WHITE ### metal ### $NC"
    echo
    cargo -q test $CARGO_EXTRA -q -p tract-metal
fi

$ROOT/api/proxy/ci.sh

# doc test are not finding libtensorflow.so
if ! cargo -q test $CARGO_EXTRA -q -p tract-tensorflow --lib $ALL_FEATURES
then
    # this crate triggers an incremental bug on nightly.
    cargo clean -p tract-tensorflow
    cargo -q test $CARGO_EXTRA -q -p tract-tensorflow --lib $ALL_FEATURES
fi
