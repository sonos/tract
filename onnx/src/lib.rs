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
extern crate protobuf;
#[allow(unused_imports)]
#[macro_use]
extern crate tract_core;
extern crate tract_linalg;

pub mod model;
pub mod ops;
pub mod pb {
    #![allow(unknown_lints)]

    #![cfg_attr(rustfmt, rustfmt_skip)]

    #![allow(box_pointers)]
    #![allow(dead_code)]
    #![allow(missing_docs)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(non_upper_case_globals)]
    #![allow(trivial_casts)]
    #![allow(unsafe_code)]
    #![allow(unused_imports)]
    #![allow(unused_results)]
    include!(concat!(env!("OUT_DIR"), "/protobuf-generated/onnx.rs"));
}
pub mod pb_helpers;
pub mod tensor;

pub use model::Onnx;
use tract_core::internal::*;

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
