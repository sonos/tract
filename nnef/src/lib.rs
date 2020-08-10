#[macro_use]
extern crate log;

pub mod ast;
pub mod deser;
pub mod framework;
pub mod ops;
pub mod ser;
pub mod registry;
pub mod tensors;

pub use ast::ProtoModel;
pub use tract_core::prelude;

pub mod internal {
    pub use crate::ast::{FragmentDecl, FragmentDef, RValue};
    pub use crate::framework::Nnef;
    pub use crate::deser::{ResolvedInvocation, ModelBuilder};
    pub use crate::registry::*;
    pub use crate::ser::{invocation, numeric, IntoAst};
    pub use std::any::TypeId;
    pub use tract_core::internal::*;
}

pub fn nnef() -> framework::Nnef {
    framework::Nnef::new()
}
