#!/bin/sh
set -e

[ -d $ROOT/.travis ] || exit 1 "\$ROOT not set correctly '$ROOT'"

if [ -z "$RUSTUP_TOOLCHAIN" ]
then
    export RUSTUP_TOOLCHAIN=1.85.0
fi

export RUSTUP_TOOLCHAIN
PATH=$PATH:$HOME/.cargo/bin

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
        if [ "$RUNNER_ENVIRONMENT" != "self-hosted" ]
        then
            if [ `whoami` != "root" ]
            then
                SUDO=sudo
            fi
            $SUDO apt-get update
            $SUDO apt-get install -y llvm python3 python3-numpy jshon wget curl build-essential sudo jshon clang
            aws --version || $SUDO apt-get install -y awscli
        fi
    fi

    which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y
    rustup update
    rustup toolchain add $RUSTUP_TOOLCHAIN

    touch /tmp/ci-setup-done
fi

S3=https://s3.amazonaws.com/tract-ci-builds/tests

if  [ "$GITHUB_WORKFLOW" = "Metal tests" -o "$GITHUB_WORKFLOW" = "CUDA tests" ] 
then
    export CACHE_FILE=$ROOT/.travis/cache_file.sh
    export MODELS=$HOME/.cache/models
    export CACHEDIR=$MODELS
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
