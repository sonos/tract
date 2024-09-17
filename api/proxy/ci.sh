#!/bin/sh

ROOT=$(dirname $(realpath $0))/../..

set -ex

cargo build --release -p tract-ffi $CARGO_EXTRA
export TRACT_DYLIB_SEARCH_PATH=$ROOT/target/release
export LD_LIBRARY_PATH=$ROOT/target/release

cd $(dirname $(realpath $0))
cargo test $CARGO_EXTRA
