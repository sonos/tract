//! # Tensorflow Deploy
//!
//! Tiny, no-nonsense, self contained, portable Tensorflow inference.
//!
//! ## Example
//!
//! ```
//! # extern crate tfdeploy;
//! # extern crate ndarray;
//! # fn main() {
//! use tfdeploy::*;
//! use tfdeploy::model::*;
//!
//! // build a simple model that just add 3 to each input component
//! let mut model = Model::default();
//!
//! let input = model.add_node("input".to_string(),
//!     Box::new(tfdeploy::ops::source::Source::default())).unwrap();
//! let three = model.add_node("three".to_string(),
//!     Box::new(tfdeploy::ops::konst::Const::new(3f32.into()))).unwrap();
//! let add = model.add_node("add".to_string(),
//!     Box::new(tfdeploy::ops::math::Add::default())).unwrap();
//!
//! model.add_edge(OutletId::new(input, 0), InletId::new(add, 0)).unwrap();
//! model.add_edge(OutletId::new(three, 0), InletId::new(add, 1)).unwrap();
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
//! While creating a model from Rust code is usefull for testing the library,
//! real-life use-cases will usually load a Tensorflow or ONNX model using
//! tfdeploy-tf or tfdeploy-onnx crates.
//!
//! For a more serious example, see [inception v3 example](https://github.com/kali/tensorflow-deploy-rust/blob/master/examples/inceptionv3.rs).

// TODO: show Plan-based API in doc instead of shortcut

extern crate bit_set;
#[cfg(feature = "blis")]
extern crate blis_src;
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate error_chain;
#[cfg(feature = "image_ops")]
extern crate image;
extern crate insideout;
#[allow(unused_imports)]
#[macro_use]
extern crate itertools;
extern crate half;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
#[allow(unused_imports)]
#[macro_use]
extern crate ndarray;
extern crate num;
#[macro_use]
extern crate maplit;
#[macro_use]
extern crate objekt;

#[cfg(feature = "serialize")]
extern crate serde;
#[cfg(test)]
extern crate simplelog;
extern crate smallvec;
#[cfg(feature = "serialize")]
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate downcast_rs;

#[macro_use]
pub mod macros;

#[macro_use]
pub mod analyser;
mod broadcast;
pub mod dim;
pub mod errors;
pub mod f16;
pub mod model;
#[macro_use]
pub mod ops;
pub mod optim;
pub mod plan;
pub mod pulse;
pub mod tensor;

pub use errors::*;

pub use analyser::types::TensorFact;
pub use dim::TDim;
pub use model::{Model, Node, TVec};
pub use plan::SimplePlan;
pub use tensor::{DatumType, Tensor};

#[cfg(test)]
#[allow(dead_code)]
fn setup_test_logger() {
    use simplelog::{Config, LevelFilter, TermLogger};
    TermLogger::init(LevelFilter::Trace, Config::default()).unwrap()
}

pub trait TfdFrom<Tf>: Sized {
    fn tfd_from(t: &Tf) -> TfdResult<Self>;
}

pub trait ToTfd<Tfd>: Sized {
    fn to_tfd(&self) -> TfdResult<Tfd>;
}

impl<PB, Tfd: TfdFrom<PB>> ::ToTfd<Tfd> for PB {
    fn to_tfd(&self) -> TfdResult<Tfd> {
        Tfd::tfd_from(self)
    }
}
