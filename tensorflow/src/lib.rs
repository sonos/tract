//! # Tract TensorFlow module
//!
//! Tiny, no-nonsense, self contained, portable inference.
//!
//! ## Example
//!
//! ```
//! # extern crate tract_core;
//! # extern crate tract_tensorflow;
//! # fn main() {
//! use tract_core::prelude::*;
//!
//! // build a simple model that just add 3 to each input component
//! let tf = tract_tensorflow::tensorflow();
//! let model = tf.model_for_path("tests/models/plus3.pb").unwrap();
//!
//! // we build an execution plan. default input and output are inferred from
//! // the model graph
//! let plan = SimplePlan::new(&model).unwrap();
//!
//! // run the computation.
//! let input = tensor1(&[1.0f32, 2.5, 5.0]);
//! let mut outputs = plan.run(tvec![input]).unwrap();
//!
//! // take the first and only output tensor
//! let mut tensor = outputs.pop().unwrap();
//!
//! assert_eq!(tensor, rctensor1(&[4.0f32, 5.5, 8.0]));
//! # }
//! ```
//!

#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate error_chain;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
#[cfg(any(test, featutre = "conform"))]
extern crate env_logger;
extern crate num_traits;
extern crate protobuf;
#[macro_use]
extern crate tract_core;
#[cfg(feature = "conform")]
extern crate tensorflow;

#[cfg(feature = "conform")]
pub mod conform;

pub mod model;
pub mod ops;
pub mod tensor;
pub mod tfpb;

pub use model::Tensorflow;
use tract_core::internal::*;

pub fn tensorflow() -> Tensorflow {
    let mut ops = crate::model::TfOpRegister::default();
    ops::register_all_ops(&mut ops);
    Tensorflow { op_register: ops }
}

#[deprecated(note = "Please use tensorflow().model_for_path(..)")]
pub fn for_path(p: impl AsRef<std::path::Path>) -> TractResult<InferenceModel> {
    tensorflow().model_for_path(p)
}

#[deprecated(note = "Please use tensorflow().model_for_read(..)")]
pub fn for_reader<R: std::io::Read>(mut r: R) -> TractResult<InferenceModel> {
    tensorflow().model_for_read(&mut r)
}

#[cfg(test)]
#[allow(dead_code)]
pub fn setup_test_logger() {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Trace).init();
}
