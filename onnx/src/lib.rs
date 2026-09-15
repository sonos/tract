//! This is tract-onnx importer.
//!
//! IF YOU'RE INTEGRATING TRACT IN AN APP, PLEASE LINK THE `tract` CRATE INSTEAD.
//!
//! Sonce 0.23, the `tract` crate presents a minimal facade which is meant to be the
//! unique entry-point to tract, while tract-onnx is an internal piece of machinery.
//!
//! There is not stability commitment on tract-onnx (or other internals) API.
//!
//! If the facade does not meet your requirements, tract maintainers are interested to
//! understand your use-case.

#![allow(clippy::len_zero)]
#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
extern crate num_integer;
#[allow(unused_imports)]
#[macro_use]
pub extern crate tract_hir;

pub mod dim_expr;
pub mod model;
pub mod ops;

#[allow(clippy::all)]
pub mod pb {
    include!("prost/onnx.rs");
}

pub mod data_resolver;
pub mod pb_helpers;
pub mod tensor;

pub use model::Onnx;

pub use tract_hir::tract_core;
pub mod prelude {
    pub use crate::onnx;
    pub use tract_hir::prelude::*;
    pub use tract_onnx_opl::WithOnnx;
}
pub use tract_onnx_opl::WithOnnx;

use tract_hir::prelude::*;

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
    Onnx { op_register: ops, ..Onnx::default() }
}
