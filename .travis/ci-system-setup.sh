#!/bin/sh

[ -d $ROOT/.travis ] || exit 1 "\$ROOT not set correctly '$ROOT'"

if [ -n "$CI" -a ! -e ".setup-done" ]
then
    which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y
    PATH=$PATH:$HOME/.cargo/bin
    rustup update
    : "${RUST_VERSION:=stable}"
    rustup toolchain add $RUST_VERSION
    rustup default $RUST_VERSION
    export RUSTUP_TOOLCHAIN=$RUST_VERSION

    if [ `uname` = "Darwin" ]
    then
        sysctl -n machdep.cpu.brand_string
        python3 --version
        brew install coreutils numpy python-setuptools jshon
        PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
        export PYTHON_BIN_PATH=python3
    else
        sudo apt-get install -y llvm python3 python3-numpy jshon
    fi
    touch .setup-done
fi

S3=https://s3.amazonaws.com/tract-ci-builds/tests

if [ -n "$CI" ]
then
    MODELS=$S3
    CACHE_FILE=true
else 
    CACHE_FILE=$ROOT/.travis/cache_file.sh
    MODELS=${MODELS:-$ROOT/.cached}
    mkdir -p $MODELS
fi
