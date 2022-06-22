#!/bin/sh

set -ex

rustup toolchain add nightly
export RUSTFLAGS=-Zsanitizer=address 
export RUSTUP_TOOLCHAIN=nightly

( cd data; cargo test -q -Zbuild-std --target x86_64-unknown-linux-gnu )
( cd linalg; cargo test -q -Zbuild-std --target x86_64-unknown-linux-gnu )
