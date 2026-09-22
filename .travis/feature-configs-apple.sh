#!/bin/sh

set -ex

# Metal is a target-conditional dependency, so the ubuntu feature-configs job
# is blind to it: off Apple, every metal combination resolves to nothing.

rustup target add aarch64-apple-darwin aarch64-apple-ios

for target in aarch64-apple-darwin aarch64-apple-ios
do
    cargo check --target $target -p tract $CARGO_EXTRA
    cargo check --target $target -p tract --no-default-features $CARGO_EXTRA
    cargo check --target $target -p tract --no-default-features --features metal $CARGO_EXTRA
    cargo check --target $target -p tract-libcli $CARGO_EXTRA
    cargo check --target $target -p tract-libcli --features metal $CARGO_EXTRA
done

# The CLI is desktop-only; check it with and without the metal dependency.
cargo check --target aarch64-apple-darwin -p tract-cli $CARGO_EXTRA
cargo check --target aarch64-apple-darwin -p tract-cli --no-default-features \
    --features transformers,pulse $CARGO_EXTRA
