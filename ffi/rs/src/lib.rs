#[macro_use]
mod macros;

pub mod dylib;
pub mod staticlib;
mod traits;

pub use traits::*;
// pub use dylib::*;
//pub use staticlib::*;
