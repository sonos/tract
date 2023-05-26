#[macro_use]
mod macros;

mod dylib;
mod staticlib;
mod traits;

pub use traits::*;
// pub use dylib::*;
//pub use staticlib::*;

pub use staticlib::{Tract, Value};
