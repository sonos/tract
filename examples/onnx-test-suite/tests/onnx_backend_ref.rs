#![allow(non_snake_case)]

mod onnx;

include!(concat!(env!("OUT_DIR"), "/tests/node.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/real.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/simple.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/pytorch-operator.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/pytorch-converted.rs"));
