//! # Tensorflow Deploy, Tensorflow module
//!
//! Tiny, no-nonsense, self contained, portable Tensorflow inference.
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
//! let tensor = tensor.take_f32s().unwrap();
//! assert_eq!(tensor, ndarray::arr1(&[4.0, 5.5, 8.0]).into_dyn());
//! # }
//! ```
//!
//! For a more serious example, see [inception v3 example](https://github.com/kali/tensorflow-deploy-rust/blob/master/examples/inceptionv3.rs).

#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate error_chain;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
extern crate ndarray;
extern crate num;
extern crate protobuf;
#[macro_use]
extern crate tract_core;

pub mod model;
pub mod ops;
pub mod optim;
pub mod tensor;
pub mod tfpb;

pub use self::model::for_path;
pub use self::model::for_reader;

pub trait ToTensorflow<Tf>: Sized {
    fn to_tf(&self) -> tract_core::TractResult<Tf>;
}
