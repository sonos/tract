use crate::internal::*;

pub(super) mod core;
pub(super) mod nnef;
pub(super) mod resource;

pub use nnef::tract_nnef;

pub fn tract_core() -> Registry {
    let mut reg = Registry::new("tract_core");
    core::register(&mut reg);
    reg
}

pub fn tract_resource() -> Registry {
    let mut reg = Registry::new("tract_resource");
    resource::register(&mut reg);
    reg
}
