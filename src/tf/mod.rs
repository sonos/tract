pub mod tfpb;
pub mod model;
pub mod tensor;

pub use self::model::for_path;
pub use self::model::for_reader;

pub trait Protobuf<Tf>: Sized {
    fn from_pb(t:&Tf) -> ::Result<Self>;
    fn to_pb(&self) -> ::Result<Tf>;
}

