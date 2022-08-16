use std::{env, fs, path};

fn main() {
    let workdir = path::PathBuf::from(env::var("OUT_DIR").unwrap()).join("prost");
    let _ = fs::create_dir_all(&workdir);
    std::env::set_var("PROTOC", protobuf_src::protoc());
    prost_build::Config::new()
        .out_dir(workdir)
        .compile_protos(&["protos/onnx/onnx.proto3"], &["protos/"])
        .unwrap();
}
