use std::io::{BufRead, Write};
use std::{env, fs, io, path};

fn main() {
    let workdir = path::PathBuf::from(env::var("OUT_DIR").unwrap()).join("prost");
    let _ = fs::create_dir_all(&workdir);
    prost_build::Config::new()
        .out_dir(workdir)
        .compile_protos(&["protos/onnx/onnx.proto3"], &["protos/"]).unwrap();
}
