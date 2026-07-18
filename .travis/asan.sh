#!/bin/sh

set -ex

# RUSTFLAGS=-Zsanitizer=address cargo +nightly test -Zbuild-std --target $(rustc -vV | sed -n 's|host: ||p')

TARGET=$(rustc -vV | sed -n 's|host: ||p')

rustup toolchain add nightly
rustup component add rust-src --toolchain nightly-$TARGET

export RUSTFLAGS=-Zsanitizer=address
export RUSTUP_TOOLCHAIN=nightly
export RUST_VERSION=nightly
export CARGO_EXTRA="--target $TARGET"

# macos ld-prime rejects inventory's asan static initializers with
# "initializer pointer has no target"; the classic linker links them fine.
if [ $(uname) == "Darwin" ]
then
    RUSTFLAGS="$RUSTFLAGS -Clink-arg=-Wl,-ld_classic"
fi
export RUSTDOCFLAGS=$RUSTFLAGS

cargo -q test -q -p tract-linalg $CARGO_EXTRA

# inventory, asan and macos liner are not playing nice, so we have to stop there 
if [ $(uname) == "Darwin" ]
then
    exit 0
fi

cargo -q test -q -p tract-core --features paranoid_assertions $CARGO_EXTRA

./.travis/regular-tests.sh
if [ -n "$CI" ]
then
    cargo clean
fi
./.travis/onnx-tests.sh
if [ -n "$CI" ]
then
    cargo clean
fi
./.travis/cli-tests.sh

if [ -n "$CI" ]
then
    cargo clean
fi

# Build libtract.so with asan, then run proxy tests against it
cargo build -p tract-ffi $CARGO_EXTRA
LIBTRACT_DIR=$(dirname $(find target -name 'libtract.so' | head -1))
TRACT_DYLIB_SEARCH_PATH=$LIBTRACT_DIR LD_LIBRARY_PATH=$LIBTRACT_DIR cargo -q test -q -p tract-proxy $CARGO_EXTRA

