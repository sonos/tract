extern crate colored;
extern crate git2;
extern crate protobuf;
extern crate tempdir;
extern crate tfdeploy;
extern crate tfdeploy_onnx;

use std::{fs, path};

use tempdir::TempDir;
use tfdeploy::*;
use tfdeploy_onnx::pb::TensorProto;
use tfdeploy_onnx::*;

pub fn load_half_dataset(prefix: &str, path: &path::Path) -> TVec<Tensor> {
    let mut vec = tvec!();
    let len = fs::read_dir(path)
        .unwrap()
        .filter(|d| {
            d.as_ref()
                .unwrap()
                .file_name()
                .to_str()
                .unwrap()
                .starts_with(prefix)
        })
        .count();
    for i in 0..len {
        let filename = path.join(format!("{}_{}.pb", prefix, i));
        let mut file = fs::File::open(filename).unwrap();
        let tensor: TensorProto = ::protobuf::parse_from_reader(&mut file).unwrap();
        vec.push(tensor.to_tfd().unwrap())
    }
    vec
}

pub fn load_dataset(path: &path::Path) -> (TVec<Tensor>, TVec<Tensor>) {
    (
        load_half_dataset("input", path),
        load_half_dataset("output", path),
    )
}

fn run_one(root: &path::Path) {
    let model = for_path(root.join("model.onnx")).unwrap();
    let inputs: Vec<&str> = model.guess_inputs().iter().map(|n| &*n.name).collect();
    let outputs: Vec<&str> = model.guess_outputs().iter().map(|n| &*n.name).collect();
    let plan = SimplePlan::new(&model, &*inputs, &*outputs).unwrap();
    for d in fs::read_dir(root).unwrap() {
        let d = d.unwrap();
        if d.metadata().unwrap().is_dir()
            && d.file_name()
                .to_str()
                .unwrap()
                .starts_with("test_data_set_")
        {
            let (inputs, expected) = load_dataset(&d.path());
            let computed = plan.run(inputs).unwrap().remove(0);
            assert_eq!(computed.len(), expected.len());
            computed
                .iter()
                .zip(expected.iter())
                .for_each(|(a, b)| assert!(a.close_enough(b, true)));
        }
    }
}

fn main() {
    use colored::Colorize;
    let dir = TempDir::new("onnx").unwrap();
    let url = "https://github.com/onnx/onnx";
    git2::Repository::clone(url, &dir).unwrap();
    let node_tests = dir.path().join("onnx/backend/test/data/node");
    let mut tests: Vec<String> = fs::read_dir(&node_tests)
        .unwrap()
        .map(|de| de.unwrap().file_name().to_str().unwrap().to_owned())
        .collect();
    tests.sort();
    let mut errors = 0;
    for test in tests {
        let path = node_tests.join(&test);
        match std::panic::catch_unwind(|| run_one(&path)) {
            Ok(()) => println!("{} {}", test, "OK".green()),
            Err(_) => {
                println!("{} {}", test, "ERROR".bright_red());
                errors += 1;
            }
        }
    }
    if errors > 0 {
        panic!("{} errors", errors);
    }
}
