#![allow(clippy::len_zero)]
//! # Tract TensorFlow module
//!
//! Tiny, no-nonsense, self contained, portable inference.
//!
//! ## Example
//!
//! ```
//! # extern crate tract_tensorflow;
//! # fn main() {
//! use tract_tensorflow::prelude::*;
//!
//! // build a simple model that just add 3 to each input component
//! let tf = tensorflow();
//! let mut model = tf.model_for_path("tests/models/plus3.pb").unwrap();
//!
//! // set input input type and shape, then optimize the network.
//! model.set_input_fact(0, f32::fact(&[3]).into()).unwrap();
//! let model = model.into_optimized().unwrap();
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
#[allow(unused_imports)]
#[macro_use]
extern crate log;
#[cfg(test)]
extern crate env_logger;
extern crate prost;
extern crate prost_types;
#[cfg(feature = "conform")]
extern crate tensorflow;
pub extern crate tract_hir;

#[cfg(feature = "conform")]
pub mod conform;

pub mod model;
pub mod ops;
pub mod tensor;
pub mod tfpb;

pub use model::Tensorflow;

pub fn tensorflow() -> Tensorflow {
    let mut ops = crate::model::TfOpRegister::default();
    ops::register_all_ops(&mut ops);
    Tensorflow { op_register: ops }
}

pub use tract_hir::tract_core;
pub mod prelude {
    pub use crate::tensorflow;
    pub use tract_hir::prelude::*;
    pub use tract_hir::tract_core;
}

#[cfg(test)]
#[allow(dead_code)]
pub fn setup_test_logger() {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Trace).init();
}
