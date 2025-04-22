#!/bin/sh
set -e

[ -d $ROOT/.travis ] || exit 1 "\$ROOT not set correctly '$ROOT'"

if [ `whoami` != "root" ]
then
    SUDO=sudo
fi

if [ -n "$CI" -a ! -e /tmp/ci-setup-done ]
then
    if [ `uname` = "Darwin" ]
    then
        sysctl -n machdep.cpu.brand_string
        python3 --version
        brew install coreutils numpy python-setuptools jshon
        PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
        export PYTHON_BIN_PATH=python3
    else
        $SUDO apt-get update
        $SUDO apt-get install -y llvm python3 python3-numpy jshon wget curl build-essential sudo jshon
        aws --version || $SUDO apt-get install -y awscli
    fi

    if [ -z "$RUST_VERSION" ]
    then
        export RUST_VERSION=1.75.0
    fi

    which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y
    PATH=$PATH:$HOME/.cargo/bin
    rustup update
    : "${RUST_VERSION:=stable}"
    rustup toolchain add $RUST_VERSION
    rustup default $RUST_VERSION
    export RUSTUP_TOOLCHAIN=$RUST_VERSION

    touch /tmp/ci-setup-done
fi

S3=https://s3.amazonaws.com/tract-ci-builds/tests

if  [ "$GITHUB_WORKFLOW" = "Metal tests" ]
then
    CACHE_FILE=$ROOT/.travis/cache_file.sh
    MODELS=$HOME/.cache/models
    CACHEDIR=$MODELS
    mkdir -p $MODELS
elif [ -n "$CI" ]
then
    MODELS=$S3
    CACHE_FILE=true
else 
    CACHE_FILE=$ROOT/.travis/cache_file.sh
    MODELS=${MODELS:-$ROOT/.cached}
    mkdir -p $MODELS
fi
