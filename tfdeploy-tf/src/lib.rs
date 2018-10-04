//! # Tensorflow Deploy, Tensorflow module
//!
//! Tiny, no-nonsense, self contained, portable Tensorflow inference.
//!
//! ## Example
//!
//! ```
//! # extern crate tfdeploy;
//! # extern crate tfdeploy_tf;
//! # extern crate ndarray;
//! # fn main() {
//! use tfdeploy::*;
//!
//! // build a simple model that just add 3 to each input component
//! let model = tfdeploy_tf::for_path("tests/models/plus3.pb").unwrap();
//!
//! // "input" and "output" are tensorflow graph node names.
//! // we build an execution plan for computing output from input
//! let plan = SimplePlan::new(&model, &["input"], &["output"]).unwrap();
//!
//! // run the computation.
//! let input = ndarray::arr1(&[1.0f32, 2.5, 5.0]);
//! let mut outputs = plan.run(tvec![input.into()]).unwrap();
//!
//! // take the tensors coming out of the only output node
//! let mut tensors = outputs.pop().unwrap();
//!
//! // grab the first (and only) tensor of this output, and unwrap it as array of f32
//! let tensor = tensors.pop().unwrap().take_f32s().unwrap();
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
extern crate tfdeploy;

pub mod tfpb;
pub mod model;
pub mod tensor;
pub mod ops;

pub use self::model::for_path;
pub use self::model::for_reader;

pub trait ToTensorflow<Tf>: Sized {
    fn to_tf(&self) -> tfdeploy::TfdResult<Tf>;
}

