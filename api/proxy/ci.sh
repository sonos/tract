#!/bin/sh

ROOT=$(dirname $(realpath $0))/../..

set -ex

cargo build --release -p tract-ffi $CARGO_EXTRA
SO=$(cargo build  --message-format=json --release -p tract-ffi $CARGO_EXTRA | grep cdylib | jshon -e  filenames -e 0 -u)
SO_PATH=$(dirname $SO)
export TRACT_DYLIB_SEARCH_PATH=$SO_PATH
export LD_LIBRARY_PATH=$SO_PATH

cd $(dirname $(realpath $0))
cargo test $CARGO_EXTRA
