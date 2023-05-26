#[macro_use]
mod macros;

//mod dylib;
mod staticlib;
mod traits;

pub use traits::*;
// pub use dylib::*;
//pub use staticlib::*;

//pub use dylib::Tract;
pub use staticlib::Tract;
