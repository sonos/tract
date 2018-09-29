#!/bin/sh

set -ex

export CI=true

cargo build --release
cargo test --release --all
cargo check --benches --all # running benches on travis is useless
