#!/bin/sh

set -ex

cargo build
cargo test
cargo check --benches # running benches on travis is useless
cargo doc
(cd tfdeploy-tf ; cargo test)
(cd tfdeploy-onnx ; env -u RUST_BACKTRACE cargo test --release)
(cd conform ; cargo test --release)
(cd exs/inceptionv3 ; cargo test --release)
(cd cli ; cargo test)
