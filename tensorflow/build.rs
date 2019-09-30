use std::io::{BufRead, Write};
use std::{env, fs, io, path};

fn main() {
    let inputs: Vec<path::PathBuf> = fs::read_dir("protos/tensorflow/core/framework")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();

    let workdir = path::PathBuf::from(env::var("OUT_DIR").unwrap()).join("protobuf-generated");
    dbg!(&workdir);
    let _ = fs::create_dir_all(&workdir);

    protobuf_codegen_pure::run(protobuf_codegen_pure::Args {
        out_dir: workdir.to_str().unwrap(),
        input: &*inputs.iter().map(|a| a.to_str().unwrap()).collect::<Vec<&str>>(),
        includes: &["protos"],
        customize: protobuf_codegen_pure::Customize { ..Default::default() },
    })
    .unwrap();

    for input in inputs {
        let mut broken = workdir.join(input.file_name().unwrap());
        let mut fixed = broken.clone();
        fixed.set_extension("rs");
        broken.set_extension("rs.orig");
        fs::rename(&fixed, &broken).unwrap();
        let mut f = fs::File::create(fixed).unwrap();
        for line in io::BufReader::new(fs::File::open(broken).unwrap()).lines() {
            let line = line.unwrap();
            if !line.starts_with("#![") && !line.starts_with("//!") {
                writeln!(f, "{}", line).unwrap();
            }
        }
    }
}
