use crate::internal::*;

mod core;
mod nnef;

pub use nnef::tract_nnef;

pub fn tract_core() -> Registry {
    let mut reg = Registry::new("tract_core");
    core::register(&mut reg);
    reg
}
