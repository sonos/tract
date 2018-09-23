extern crate colored;
#[macro_use]
extern crate error_chain;
extern crate git2;
extern crate protobuf;
extern crate tempdir;
extern crate tfdeploy;
extern crate tfdeploy_onnx;

use std::{fs, path};

mod onnx;

#[test]
fn node() {
    onnx::ensure_onnx_git_checkout().unwrap();
    let dir = path::PathBuf::from(onnx::ONNX_DIR);
    let node_tests = dir.join("onnx/backend/test/data/node");
    let mut tests: Vec<String> = fs::read_dir(&node_tests)
        .unwrap()
        .map(|de| de.unwrap().file_name().to_str().unwrap().to_owned())
        .collect();
    tests.sort();
    let mut errors = 0;
    for test in tests {
        if onnx::run_one(&node_tests, &test).is_ok() {
            errors += 1
        }
    }
    if errors != 0 {
        panic!("{} errors", errors)
    }
}
