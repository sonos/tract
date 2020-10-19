//! # Tract
//!
//! Tiny, no-nonsense, self contained, portable TensorFlow and ONNX inference.
//!
//! ## Example
//!
//! ```
//! # extern crate tract_core;
//! # fn main() {
//! use tract_core::internal::*;
//!
//! // build a simple model that just add 3 to each input component
//! let mut model = TypedModel::default();
//!
//! let input_fact = TypedFact::dt_shape(f32::datum_type(), [3].as_ref()).unwrap();
//! let input = model.add_source("input", input_fact).unwrap();
//! let three = model.add_const("three".to_string(), tensor1(&[3f32])).unwrap();
//! let add = model.wire_node("add".to_string(),
//!     tract_core::ops::math::add::bin_typed(),
//!     [input, three].as_ref()
//!     ).unwrap();
//!
//! model.auto_outputs().unwrap();
//!
//! // We build an execution plan. Default inputs and outputs are inferred from
//! // the model graph.
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
//! While creating a model from Rust code is useful for testing the library,
//! real-life use-cases will usually load a TensorFlow or ONNX model using
//! tract-tensorflow or tract-onnx crates.
//!

pub extern crate anyhow;
extern crate bit_set;
#[macro_use]
extern crate derive_new;
#[macro_use]
pub extern crate downcast_rs;
#[macro_use]
extern crate educe;
#[allow(unused_imports)]
#[macro_use]
pub extern crate itertools;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
//#[macro_use]
//extern crate maplit;
#[allow(unused_imports)]
#[macro_use]
pub extern crate ndarray;
#[cfg(test)]
extern crate env_logger;
extern crate num_integer;
pub extern crate num_traits;
#[cfg(test)]
extern crate proptest;
extern crate smallvec;

pub extern crate tract_data;
pub extern crate tract_linalg;

//#[macro_use]
//pub mod macros;
#[macro_use]
pub mod ops;

pub mod broadcast;
//pub mod datum;
//pub mod dim;
pub mod framework;
mod hash;
pub mod model;
mod optim;
pub mod plan;
//pub mod tensor;

pub use dyn_clone;

pub type TractError = anyhow::Error;
pub type TractResult<T> = Result<T, anyhow::Error>;

/// This prelude is meant for code using tract.
pub mod prelude {
    pub use crate::framework::Framework;
    pub use crate::model::*;
    pub use crate::plan::{SimplePlan, SimpleState};
    pub use crate::{TractError, TractResult};
    pub use std::sync::Arc;
    pub use tract_data::prelude::*;

    pub use itertools as tract_itertools;
    pub use ndarray as tract_ndarray;
    pub use num_traits as tract_num_traits;
    pub use tract_linalg;
    pub use tract_data;
}

/// This prelude is meant for code extending tract (like implementing new ops).
pub mod internal {
    pub use crate::model::*;
    pub use crate::ops::change_axes::*;
    pub use crate::ops::element_wise::ElementWiseMiniOp;
    pub use crate::ops::invariants::*;
    pub use crate::ops::{AxisInfo, Cost, EvalOp, Invariants, Op, OpState, Validation};
    pub use crate::plan::SessionState;
    pub use crate::prelude::*;
    pub use anyhow::{bail, format_err, Context as TractErrorContext};
    pub use downcast_rs as tract_downcast_rs;
    pub use std::borrow::Cow;
    pub use std::collections::HashMap;
    pub use std::hash::Hash;
    pub use std::marker::PhantomData;
    pub use tract_data::internal::*;
    pub use tract_data::{
        dispatch_copy, dispatch_datum, dispatch_datum_by_size, dispatch_floatlike, dispatch_numbers,
    };
    pub use tvec;
    pub use {args_1, args_2, args_3, args_4, args_5, args_6, args_7, args_8};
    pub use {as_op, impl_op_same_as, not_a_typed_op, op_as_typed_op};
    pub use {bin_to_super_type, element_wise, element_wise_oop};
}

#[cfg(test)]
#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}
