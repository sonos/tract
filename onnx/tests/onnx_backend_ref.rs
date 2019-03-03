#![allow(non_snake_case)]
extern crate env_logger;
extern crate flate2;
extern crate fs2;
#[macro_use]
extern crate log;
extern crate mio_httpc;
extern crate protobuf;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate tar;
extern crate tract_core;
extern crate tract_onnx;

mod onnx;

include!(concat!(env!("OUT_DIR"), "/tests/node.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/real.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/simple.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/pytorch-operator.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/pytorch-converted.rs"));
