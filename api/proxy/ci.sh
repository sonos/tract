#!/bin/sh

ROOT=$(dirname $(realpath $0))/../..

set -ex

cargo build --release -p tract-ffi
export TRACT_DYLIB_SEARCH_PATH=$ROOT/target/release
export LD_LIBRARY_PATH=$ROOT/target/release
cargo test
