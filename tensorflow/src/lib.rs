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
extern crate prost;
extern crate prost_types;
#[macro_use]
extern crate tract_core;
#[cfg(feature = "conform")]
extern crate tensorflow;
/*
mod google {
    mod protobuf {
        include!(concat!(env!("OUT_DIR"), "/prost/google.protobuf.rs"));
    }
}

mod tensorflow {
    include!(concat!(env!("OUT_DIR"), "/prost/tensorflow.rs"));
}
*/

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

#[cfg(test)]
#[allow(dead_code)]
pub fn setup_test_logger() {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Trace).init();
}
