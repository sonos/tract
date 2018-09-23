extern crate colored;
#[macro_use]
extern crate error_chain;
extern crate git2;
extern crate protobuf;
extern crate tempdir;
extern crate tfdeploy;
extern crate tfdeploy_onnx;

mod onnx;

#[test]
fn node() {
    onnx::run_all("node")
}

#[test]
fn simple() {
    onnx::run_all("simple")
}
