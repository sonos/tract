#!/bin/sh

WHITE='\033[1;37m'
NC='\033[0m' # No Color

if [ -e /proc/cpuinfo ]
then
    grep "^flags" /proc/cpuinfo | head -1 | \
        grep --color=always '\(s\?sse[0-9_]*\|fma\|f16c\|avx[^ ]*\)'
fi

set -x

. $(dirname $0)/ci-system-setup.sh

set -e

if [ `arch` = "x86_64" -a "$RUST_VERSION" = "stable" ]
then
    ALL_FEATURES=--all-features
fi

set +x

for c in data linalg core nnef hir onnx pulse onnx-opl pulse-opl rs proxy
do
    echo
    echo "$WHITE ### $c ### $NC"
    echo
    cargo -q test $CARGO_EXTRA -q -p tract-$c
done

for c in test-rt/test*
do
    if [ "$c" != "test-rt/test-tflite" ]
    then
        echo
        echo "$WHITE ### $c ### $NC"
        echo
        (cd $c; cargo test -q $CARGO_EXTRA)
    fi
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

# if [ -n "$GITHUB_ACTIONS" ]
# then
#     CLEANUP="rm -rf $CACHEDIR/*"
# else
     CLEANUP=true
# fi

$CLEANUP
cargo -q test $CARGO_EXTRA -q -p tract-rs
$CLEANUP
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p core-proptest-pulse $ALL_FEATURES
$CLEANUP
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p lstm-proptest-onnx-vs-tf $ALL_FEATURES
$CLEANUP
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p nnef-inceptionv3 $ALL_FEATURES
$CLEANUP
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-inceptionv3 $ALL_FEATURES
$CLEANUP
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-mobilenet-v2 $ALL_FEATURES
$CLEANUP
cargo -q test $CARGO_EXTRA -q --profile opt-no-lto -p tf-moz-deepspeech $ALL_FEATURES
CACHEDIR=$OLD_CACHEDIR
