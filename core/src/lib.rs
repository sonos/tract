//! # Tract
//!
//! Tiny, no-nonsense, self contained, portable SharedTensor and ONNX inference.
//!
//! ## Example
//!
//! ```
//! # extern crate tract_core;
//! # extern crate ndarray;
//! # fn main() {
//! use tract_core::*;
//! use tract_core::model::*;
//! use tract_core::model::dsl::*;
//!
//! // build a simple model that just add 3 to each input component
//! let mut model = Model::default();
//!
//! let input = model.add_source("input").unwrap();
//! let three = model.add_const("three".to_string(), 3f32.into()).unwrap();
//! let add = model.add_node("add".to_string(),
//!     Box::new(tract_core::ops::math::Add::default())).unwrap();
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
//! let tensor = tensor.to_array_view::<f32>().unwrap();
//! assert_eq!(tensor, ndarray::arr1(&[4.0, 5.5, 8.0]).into_dyn());
//! # }
//! ```
//!
//! While creating a model from Rust code is usefull for testing the library,
//! real-life use-cases will usually load a SharedTensor or ONNX model using
//! tract-tf or tract-onnx crates.
//!

// TODO: show Plan-based API in doc instead of shortcut

extern crate bit_set;
#[cfg(feature = "blis")]
extern crate blis_src;
#[macro_use]
extern crate custom_debug_derive;
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate downcast_rs;
#[macro_use]
extern crate error_chain;
#[cfg(feature = "image_ops")]
extern crate image;
extern crate insideout;
#[allow(unused_imports)]
#[macro_use]
extern crate itertools;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
#[allow(unused_imports)]
#[macro_use]
pub extern crate ndarray;
extern crate num_integer;
extern crate num_traits;
#[macro_use]
extern crate maplit;
#[cfg(test)]
extern crate matrixmultiply;
#[macro_use]
extern crate objekt;
#[cfg(test)]
extern crate proptest;
#[cfg(feature = "serialize")]
extern crate serde;
#[cfg(test)]
extern crate simplelog;
extern crate smallvec;
#[cfg(feature = "serialize")]
#[macro_use]
extern crate serde_derive;

extern crate tract_linalg;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod analyser;
#[macro_use]
pub mod ops;

pub mod broadcast;
pub mod context;
pub mod datum;
pub mod dim;
pub mod errors;
pub mod model;
mod ndarray_dummy_packed_mm;
pub mod optim;
pub mod plan;
pub mod pulse;
pub mod tensor;

pub use crate::errors::*;

pub use crate::analyser::types::TensorFact;
pub use crate::datum::DatumType;
pub use crate::dim::TDim;
pub use crate::model::{Model, Node, TVec};
pub use crate::plan::{SimplePlan, SimpleState};
pub use crate::tensor::{SharedTensor, Tensor};

#[cfg(test)]
#[allow(dead_code)]
fn setup_test_logger() {
    use simplelog::{Config, LevelFilter, TermLogger};
    TermLogger::init(LevelFilter::Trace, Config::default()).unwrap()
}

pub trait Tractify<Other>: Sized {
    fn tractify(t: &Other) -> TractResult<Self>;
}

pub trait ToTract<Tract>: Sized {
    fn tractify(&self) -> TractResult<Tract>;
}

impl<PB, Tract: Tractify<PB>> crate::ToTract<Tract> for PB {
    fn tractify(&self) -> TractResult<Tract> {
        Tract::tractify(self)
    }
}
