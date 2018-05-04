extern crate protoc_rust;

use std::{fs, io, path};

use std::io::{BufRead, Write};

fn main() {
    let inputs: Vec<path::PathBuf> = fs::read_dir("tensorflow/core/framework")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();

    let _ = fs::create_dir_all("target/generated");

    protoc_rust::run(protoc_rust::Args {
        out_dir: "target/generated",
        input: &*inputs
            .iter()
            .map(|a| a.to_str().unwrap())
            .collect::<Vec<&str>>(),
        includes: &["."],
    }).expect("protoc");

    for input in inputs {
        let mut broken = path::PathBuf::from("target/generated").join(input.file_name().unwrap());
        let mut fixed = broken.clone();
        fixed.set_extension("rs");
        broken.set_extension("rs.orig");
        fs::rename(&fixed, &broken).unwrap();
        let mut f = fs::File::create(fixed).unwrap();
        for line in io::BufReader::new(fs::File::open(broken).unwrap()).lines() {
            let line = line.unwrap();
            if !line.starts_with("#![") {
                writeln!(f, "{}", line).unwrap();
            }
        }
    }
}
