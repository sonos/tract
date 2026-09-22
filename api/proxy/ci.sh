#!/bin/sh

ROOT=$(dirname $(realpath $0))/../..

set -ex

cargo build --release -p tract-ffi $CARGO_EXTRA
SO=$(cargo build --message-format=json --release -p tract-ffi $CARGO_EXTRA | jq -r 'select(.target.kind[]? == "cdylib") | .filenames[0]' | head -1)
SO_PATH=$(dirname $SO)
export TRACT_DYLIB_SEARCH_PATH=$SO_PATH
export LD_LIBRARY_PATH=$SO_PATH
export DYLD_LIBRARY_PATH=$SO_PATH

cd $(dirname $(realpath $0))
cargo test $CARGO_EXTRA
