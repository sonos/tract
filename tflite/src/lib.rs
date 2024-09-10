#![allow(dead_code)]
#[macro_use]
extern crate derive_new;

mod model;
mod ops;
mod registry;
pub mod rewriter;
mod ser;
mod tensors;

#[allow(
    unused_imports,
    clippy::extra_unused_lifetimes,
    clippy::missing_safety_doc,
    clippy::derivable_impls,
    clippy::needless_lifetimes,
    clippy::too_long_first_doc_paragraph
)]
mod tflite_generated;
pub use tflite_generated::tflite;

pub use model::Tflite;

pub mod prelude {
    pub use tract_core::prelude::*;
}

pub mod internal {
    pub use crate::model::TfliteProtoModel;
    pub use tract_core::internal::*;
    pub use tract_core;
}

pub fn tflite() -> Tflite {
    Tflite::default()
}
