#![allow(clippy::len_zero)]
#[macro_use]
extern crate log;

pub mod ast;
pub mod deser;
pub mod framework;
pub mod ops;
pub mod registry;
pub mod resource;
pub mod ser;
pub mod tensors;

pub use ast::ProtoModel;

pub use tract_core;
pub use tract_core::prelude::tract_ndarray;
pub use tract_core::prelude::tract_num_traits;

pub mod prelude {
    pub use tract_core;
    pub use tract_core::prelude::*;
}

pub mod internal {
    pub use crate::ast::parse::parse_parameters;
    pub use crate::ast::dump_doc::DocDumper;
    pub use crate::ast::{
        param, FragmentDecl, FragmentDef, Identifier, Parameter, RValue, TypeName,
    };
    pub use crate::deser::{ModelBuilder, ResolvedInvocation, Value};
    pub use crate::framework::Nnef;
    pub use crate::prelude::*;
    pub use crate::registry::*;
    pub use crate::resource::{
        DatLoader, GraphNnefLoader, GraphQuantLoader, Resource, ResourceLoader, TypedModelResource, TypedModelLoader,
    };
    pub use crate::ser::{invocation, logical, numeric, string, IntoAst};
    pub use std::any::TypeId;
    pub use tract_core::internal::*;
}

pub fn nnef() -> framework::Nnef {
    framework::Nnef::default()
}
