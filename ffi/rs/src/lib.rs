#[macro_use]
mod macros;

// #[cfg(feature="dylib")]
pub mod dylib;

// #[cfg(feature="staticlib")]
pub mod staticlib;
mod traits;

pub use traits::*;
// pub use dylib::*;
//pub use staticlib::*;
