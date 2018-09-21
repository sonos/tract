#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate error_chain;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
extern crate ndarray;
extern crate num;
extern crate protobuf;
#[macro_use]
extern crate tfdeploy;

pub mod tfpb;
pub mod model;
pub mod tensor;
pub mod ops;

pub use self::model::for_path;
pub use self::model::for_reader;

pub trait ToTensorflow<Tf>: Sized {
    fn to_tf(&self) -> ::tfdeploy::Result<Tf>;
}

