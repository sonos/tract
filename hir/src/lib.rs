#![allow(clippy::len_zero)]
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate log;

#[macro_use]
pub mod macros;
pub mod framework;

pub mod infer;

pub extern crate tract_core;

pub use tract_core::prelude::tract_ndarray;
pub use tract_core::prelude::tract_num_traits;

pub mod ops;

pub mod prelude {
    pub use crate::infer::InferenceFact;
    pub use crate::infer::InferenceModel;
    pub use crate::infer::InferenceModelExt;
    pub use crate::infer::InferenceSimplePlan;
    pub use tract_core::prelude::*;
}

pub mod internal {
    pub use super::prelude::*;
    pub use crate::infer::*;
    pub use crate::ops::binary::BinIntoHir;
    pub use crate::ops::element_wise::ElementWiseIntoHir;
    pub use crate::ops::expandable::{expand, inference_wrap, Expansion};
    pub use tract_core;
    pub use tract_core::internal::*;
    pub use {shapefactoid, to_typed};
}

#[cfg(test)]
#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}
