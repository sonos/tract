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
for c in test-rt/test*; do
    case "$c" in
        test-rt/test-tflite)
            echo "$WHITE ### $c ### IGNORED $NC"
            continue
            ;;
        test-rt/test-metal)
            if [ "$(uname)" != "Darwin" ] || [ -n "$CI" ]; then
                echo "$WHITE ### $c ### IGNORED $NC"
                continue
            fi
            ;;
        test-rt/test-cuda)
            if ! command -v nvcc >/dev/null; then
                echo "$WHITE ### $c ### IGNORED $NC"
                continue
            fi
            ;;
    esac

    echo
    echo "$WHITE ### $c ### $NC"
    echo
    (cd "$c" && cargo test -q $CARGO_EXTRA)

    if [ -n "$CI" ]; then
        df -h
        cargo clean
    fi
done

