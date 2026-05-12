//! Compile vendored coremltools .proto files into Rust types via protox + prost-build.
//! No external `protoc` binary required.
//!
//! See `proto/MIL_PROTO_VERSION.md` for the pinned upstream commit.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir: PathBuf = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("proto");

    let proto_files: Vec<PathBuf> = fs::read_dir(&proto_dir)?
        .filter_map(|r| r.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "proto"))
        .collect();

    println!("cargo:rerun-if-changed={}", proto_dir.display());
    for p in &proto_files {
        println!("cargo:rerun-if-changed={}", p.display());
    }

    let fds = protox::compile(&proto_files, [&proto_dir])?;
    // Strip generated doc comments — coremltools' .proto files contain code-like
    // examples that prost-build emits as Rust doctests, none of which are valid
    // Rust. (Generates >100 spurious `cargo test` failures otherwise.)
    prost_build::Config::new().disable_comments(["."]).compile_fds(fds)?;

    Ok(())
}
