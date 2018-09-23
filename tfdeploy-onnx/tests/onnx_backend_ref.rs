extern crate colored;
#[macro_use]
extern crate error_chain;
extern crate flate2;
extern crate fs2;
extern crate git2;
extern crate mio_httpc;
extern crate protobuf;
extern crate rayon;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate tar;
extern crate tempdir;
extern crate tfdeploy;
extern crate tfdeploy_onnx;

mod onnx;

#[test]
fn node() {
    onnx::run_all("node")
}

#[test]
fn real() {
    onnx::run_all("real")
}

#[test]
fn simple() {
    onnx::run_all("simple")
}
