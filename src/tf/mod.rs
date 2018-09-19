pub mod tfpb;
pub mod model;
pub mod tensor;

pub use self::model::for_path;
pub use self::model::for_reader;

pub trait ToTensorflow<Tf>: Sized {
    fn to_tf(&self) -> ::Result<Tf>;
}

