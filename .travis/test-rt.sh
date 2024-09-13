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

cd $ROOT
for c in test-rt/test*
do
    if [ "$c" = "test-rt/test-tflite" ]
    then
        echo "$WHITE ### $c ### IGNORED $NC"
    elif [ "$c" = "test-rt/test-metal" -a  \( `uname` != "Darwin" -o -n "$CI" \) ]
    then
        echo "$WHITE ### $c ### IGNORED $NC"
    else
        echo
        echo "$WHITE ### $c ### $NC"
        echo
        (cd $c; cargo test -q $CARGO_EXTRA)
        if [ -n "$CI" ]
        then
            df -h
            cargo clean
        fi
    fi
done

