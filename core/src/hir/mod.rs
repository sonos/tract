#[macro_use]
pub mod macros;

pub mod array;
pub mod binary;
pub mod dummy;
pub mod element_wise;
pub mod cnn;
pub mod downsample;
pub mod identity;
pub mod framework;
pub mod konst;
pub mod logic;
pub mod matmul;
pub mod model;
pub mod nn;
pub mod unimpl;
pub mod scan;
pub mod source;

pub use framework::Framework;
