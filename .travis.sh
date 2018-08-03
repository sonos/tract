#!/bin/sh

set -ex

cargo build
cargo test
cargo check --benches # running benches on travis is useless
cargo doc
(cd conform ; cargo test)
(cd exs/inceptionv3 ; cargo test --release)
(cd cli ; cargo test)
