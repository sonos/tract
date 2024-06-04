#!/bin/sh

if [ -n "$CI" -a -e ".setup-done" ]
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
        brew install coreutils python3 numpy python-setuptools
        PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
        export PYTHON_BIN_PATH=python3
    else
        sudo apt-get install -y llvm python3 python3-numpy
    fi
    if [ -z "$CACHEDIR" ]
    then
        CACHEDIR=$(realpath `dirname $0`/../.cached)
    fi

    export CACHEDIR

    touch .setup-done
fi


