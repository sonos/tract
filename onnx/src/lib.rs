#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate error_chain;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
extern crate num_integer;
extern crate num_traits;
#[allow(unused_imports)]
#[macro_use]
extern crate tract_core;
extern crate tract_linalg;

pub mod model;
pub mod ops;

pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/prost/onnx.rs"));
}

pub mod pb_helpers;
pub mod tensor;

pub use model::Onnx;
use tract_core::internal::*;
use tract_core::infer::*;

#[deprecated(note = "Please use onnx().model_for_path(..)")]
pub fn for_path(p: impl AsRef<std::path::Path>) -> TractResult<InferenceModel> {
    onnx().model_for_path(p)
}

#[deprecated(note = "Please use onnx().model_for_read(..)")]
pub fn for_reader<R: std::io::Read>(mut r: R) -> TractResult<InferenceModel> {
    onnx().model_for_read(&mut r)
}

pub fn onnx() -> Onnx {
    let mut ops = crate::model::OnnxOpRegister::default();
    ops::register_all_ops(&mut ops);
    Onnx { op_register: ops }
}
