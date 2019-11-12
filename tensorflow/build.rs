use std::io::{BufRead, Write};
use std::{env, fs, io, path};

fn main() -> std::io::Result<()> {
    let mut inputs: Vec<path::PathBuf> = vec![];

    for dir in &[
        "protos/google/protobuf",
        "protos/tensorflow/core/framework",
        "protos/tensorflow/core/protobuf",
    ] {
        for pb in fs::read_dir(dir)? {
            inputs.push(pb?.path())
        }
    }

    let gen = path::PathBuf::from(env::var("OUT_DIR").unwrap()).join("prost");
    let _ = fs::create_dir_all(&gen);
    prost_build::Config::new()
        .out_dir(gen)

        .compile_protos(&inputs, &[path::PathBuf::from("protos/")])?;

    /*
    let raw = path::PathBuf::from(env::var("OUT_DIR").unwrap()).join("protobuf-generated-raw");
    let fixed = path::PathBuf::from(env::var("OUT_DIR").unwrap()).join("protobuf-generated");
    let _ = fs::create_dir_all(&raw);
    let _ = fs::create_dir_all(&fixed);

    protobuf_codegen_pure::run(protobuf_codegen_pure::Args {
        out_dir: raw.to_str().unwrap(),
        input: &*inputs.iter().map(|a| a.to_str().unwrap()).collect::<Vec<&str>>(),
        includes: &["protos"],
        customize: protobuf_codegen_pure::Customize { ..Default::default() },
    })
    .unwrap();

    for input in fs::read_dir(&raw)? {
        let input = input?;
        let fixed = fixed.join(input.file_name());
        let mut f = fs::File::create(fixed).unwrap();
        for line in io::BufReader::new(fs::File::open(input.path()).unwrap()).lines() {
            let line = line.unwrap();
            if !line.starts_with("#![") && !line.starts_with("//!") {
                writeln!(f, "{}", line).unwrap();
            }
        }
    }
    */
    Ok(())
}
