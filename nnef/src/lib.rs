#[macro_use]
extern crate log;

pub mod ast;
pub mod framework;
pub mod model;
pub mod ops;
pub mod ser;
pub mod tensors;

pub use tract_core::prelude;
pub use ast::ProtoModel;

pub mod internal {
    pub use std::any::TypeId;
    pub use tract_core::internal::*;
    pub use crate::ast::{FragmentDecl, FragmentDef, RValue};
    pub use crate::framework::Nnef;
    pub use crate::ops::{Registry};
    pub use crate::model::{ModelBuilder, AugmentedInvocation};
    pub use crate::ser::{ IntoAst, invocation, numeric };
}

pub fn nnef() -> framework::Nnef {
    framework::Nnef::new()
}
