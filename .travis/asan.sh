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

# asan's global redzones break the __DATA,__mod_init_func entries `inventory`
# writes: ld-prime rejects them with "initializer pointer has no target", and
# the classic linker takes them and silently registers nothing, which leaves
# every registry empty and the tests reading them vacuously green. Leave
# globals uninstrumented; heap and stack checking are unaffected.
if [ $(uname) == "Darwin" ]
then
    RUSTFLAGS="$RUSTFLAGS -Cllvm-args=-asan-globals=0"
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

# Build the dylib with asan, then run the proxy tests against it. cargo names the
# file it wrote: the extension and the profile directory both vary, and a search
# of target/ finds whatever another profile left there.
LIBTRACT=$(cargo build --message-format=json -p tract-ffi $CARGO_EXTRA \
    | jq -r 'select(.target.kind[]? == "cdylib") | .filenames[0]' | head -1)
LIBTRACT_DIR=$(dirname $LIBTRACT)
TRACT_DYLIB_SEARCH_PATH=$LIBTRACT_DIR LD_LIBRARY_PATH=$LIBTRACT_DIR \
    DYLD_LIBRARY_PATH=$LIBTRACT_DIR cargo -q test -q -p tract-proxy $CARGO_EXTRA

