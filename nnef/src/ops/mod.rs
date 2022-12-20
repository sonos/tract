use crate::internal::*;

pub(super) mod core;
pub mod nnef;
pub(super) mod resource;

pub use nnef::tract_nnef;

pub fn tract_core() -> Registry {
    let mut reg = Registry::new("tract_core")
        .with_doc("Extension `tract_core` exposes NNEF fragments for using")
        .with_doc("operator defined by tract-core crate.")
        .with_doc("")
        .with_doc("Add `extension tract_core` to `graph.nnef`");
    core::register(&mut reg);
    reg
}

pub fn tract_resource() -> Registry {
    let mut reg = Registry::new("tract_resource")
        .with_doc("Extension `tract_resource` exposes NNEF fragments for accessing")
        .with_doc("resources files in NNEF folder or archive.")
        .with_doc("")
        .with_doc("Add `extension tract_resource` to `graph.nnef`");
    resource::register(&mut reg);
    reg
}
