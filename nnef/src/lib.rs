pub mod ast;
pub mod container;
pub mod model;
pub mod ops;
pub mod ser;
pub mod tensors;


pub use model::ProtoModel;
pub use tract_core::prelude;

pub use container::load;
pub use container::save_to_dir;
pub use container::save_to_tgz;
pub use container::save;

pub fn nnef() -> model::Framework {
    model::Framework::new()
}
