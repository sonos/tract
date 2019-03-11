//! # SharedTensor Deploy, SharedTensor module
//!
//! Tiny, no-nonsense, self contained, portable SharedTensor inference.
//!
//! ## Example
//!
//! ```
//! # extern crate tract_core;
//! # extern crate tract_tensorflow;
//! # extern crate ndarray;
//! # fn main() {
//! use tract_core::*;
//!
//! // build a simple model that just add 3 to each input component
//! let model = tract_tensorflow::for_path("tests/models/plus3.pb").unwrap();
//!
//! // we build an execution plan. default input and output are inferred from
//! // the model graph
//! let plan = SimplePlan::new(&model).unwrap();
//!
//! // run the computation.
//! let input = ndarray::arr1(&[1.0f32, 2.5, 5.0]);
//! let mut outputs = plan.run(tvec![input.into()]).unwrap();
//!
//! // take the first and only output tensor
//! let mut tensor = outputs.pop().unwrap();
//!
//! // unwrap it as array of f32
//! let tensor = tensor.to_array_view::<f32>().unwrap();
//! assert_eq!(tensor, ndarray::arr1(&[4.0, 5.5, 8.0]).into_dyn());
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
extern crate ndarray;
extern crate num_traits;
extern crate protobuf;
#[cfg(any(test, featutre="conform"))]
extern crate env_logger;
#[macro_use]
extern crate tract_core;
#[cfg(feature = "conform")]
extern crate tensorflow;

#[cfg(feature = "conform")]
pub mod conform;

pub mod model;
pub mod ops;
mod optim;
pub mod tensor;
pub mod tfpb;

/*
pub use self::model::for_path;
pub use self::model::for_reader;
*/

pub trait ToSharedTensor<Tf>: Sized {
    fn to_tf(&self) -> tract_core::TractResult<Tf>;
}

use crate::tfpb::node_def::NodeDef;
use crate::tfpb::graph::GraphDef;
type Tensorflow = tract_core::model::Framework<NodeDef, GraphDef>;

pub fn tensorflow() -> Tensorflow {
    let ops = std::collections::HashMap::default();
    let mut fw = tract_core::model::Framework {
        ops,
        model_builder: Box::new(model::build),
        model_loader: Box::new(model::load),
    };
    ops::register_all_ops(&mut fw);
    fw
}

#[cfg(test)]
#[allow(dead_code)]
pub fn setup_test_logger() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Trace)
        .init();
}
