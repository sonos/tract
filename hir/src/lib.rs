pub extern crate tract_core;

pub use tract_core::hir::ops;

pub mod prelude {
    pub use tract_core::hir::prelude::*;
}

pub mod internal {
    pub use tract_core::hir::internal::*;
}
