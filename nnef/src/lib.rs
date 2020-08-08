pub mod ast;
pub mod container;
pub mod model;
pub mod ops;
pub mod ser;
pub mod tensors;


pub use model::ProtoModel;
pub use tract_core::prelude;

pub fn nnef() -> model::Framework {
    model::Framework::new()
}
