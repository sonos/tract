#!/bin/sh

ROOT=$(dirname $(realpath $0))/../..

set -ex

case "$CARGO_EXTRA" in *--profile*) PROFILE= ;; *) PROFILE=--release ;; esac

cargo build $PROFILE -p tract-ffi $CARGO_EXTRA
SO=$(cargo build --message-format=json $PROFILE -p tract-ffi $CARGO_EXTRA | jq -r 'select(.target.kind[]? == "cdylib") | .filenames[0]' | head -1)
SO_PATH=$(dirname $SO)
export TRACT_DYLIB_SEARCH_PATH=$SO_PATH
export LD_LIBRARY_PATH=$SO_PATH
export DYLD_LIBRARY_PATH=$SO_PATH

cd $(dirname $(realpath $0))
cargo test $CARGO_EXTRA
