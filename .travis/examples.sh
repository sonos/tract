#!/bin/sh

WHITE='\033[1;37m'
NC='\033[0m' # No Color

set -e

which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y
rustup update

PATH=$PATH:$HOME/.cargo/bin

: "${RUST_VERSION:=stable}"
rustup toolchain add $RUST_VERSION
rustup default $RUST_VERSION

for t in `find examples -name ci.sh`
do
    df -h
    ex=$(dirname $t)
    echo ::group:: $ex
    echo $WHITE $ex $NC
    ( cd $ex ; sh ./ci.sh )
    echo ::endgroup::
done

