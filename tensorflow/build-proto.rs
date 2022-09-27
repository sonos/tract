use std::{env, fs, path};

fn main() -> std::io::Result<()> {
    env::set_var("PROTOC", protobuf_src::protoc());
    let inputs: Vec<path::PathBuf> = {
        let mut inputs: Vec<path::PathBuf> = vec![];

        for dir in &["protos/tensorflow/core/framework", "protos/tensorflow/core/protobuf"] {
            for pb in fs::read_dir(dir)? {
                inputs.push(pb?.path())
            }
        }

        inputs.sort();
        inputs
    };

    let gen = "src/prost";
    let _ = fs::create_dir_all(&gen);
    prost_build::Config::new()
        .out_dir(gen)
        .compile_protos(&inputs, &[path::PathBuf::from("protos/")])?;

    Ok(())
}
