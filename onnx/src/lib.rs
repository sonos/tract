#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate error_chain;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
extern crate ndarray;
extern crate num_integer;
extern crate num_traits;
extern crate protobuf;
#[allow(unused_imports)]
#[macro_use]
extern crate tract_core;
extern crate tract_linalg;

pub mod model;
pub mod ops;
pub mod pb;
pub mod pb_helpers;
pub mod tensor;

pub use model::Onnx;
use tract_core::{ Framework, Model, TractResult };

#[deprecated(note="Please use onnx().model_for_path(..)")]
pub fn for_path(p: impl AsRef<std::path::Path>) -> TractResult<Model> {
    onnx().model_for_path(p)
}

#[deprecated(note="Please use onnx().model_for_read(..)")]
pub fn for_reader<R: std::io::Read>(mut r: R) -> TractResult<Model> {
    onnx().model_for_read(&mut r)
}


pub fn onnx() -> Onnx {
    let mut ops = tract_core::framework::OpRegister::default();
    ops::register_all_ops(&mut ops);
    Onnx { op_register: ops }
}

