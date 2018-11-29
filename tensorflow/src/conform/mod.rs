#![allow(unused)]
#![allow(non_snake_case)]

error_chain! {
    foreign_links {
        Io(::std::io::Error);
        NdarrayShape(::ndarray::ShapeError);
        Protobuf(::protobuf::ProtobufError);
        StrUtf8(::std::str::Utf8Error);
    }
    errors {
        TFString {}
    }
}

impl ::std::convert::From<::tensorflow::Status> for Error {
    fn from(tfs: ::tensorflow::Status) -> Error {
        format!("SharedTensor error: {:?}", tfs).into()
    }
}

pub mod tf;

pub use protobuf::Message;

use tfpb;
use tfpb::tensor_shape::TensorShapeProto;
use tfpb::types::DataType;
use tract_core::ops::prelude::*;

pub fn placeholder<Shape: Into<Option<TensorShapeProto>>>(
    name: &str,
    t: DataType,
    shape: Shape,
) -> tfpb::node_def::NodeDef {
    let mut node = tfpb::node().name(name).op("Placeholder").attr("dtype", t);
    if let Some(shape) = shape.into() {
        node = node.attr("shape", shape)
    }
    node
}

pub fn tensor_shape(dims: &[usize]) -> TensorShapeProto {
    use tfpb::tensor_shape::*;
    let mut shape = TensorShapeProto::new();
    shape.set_dim(
        dims.iter()
            .map(|&d| {
                let mut dim = TensorShapeProto_Dim::new();
                dim.set_size(d as i64);
                dim
            }).collect(),
    );
    shape
}

pub fn placeholder_f32(name: &str) -> tfpb::node_def::NodeDef {
    placeholder(name, DataType::DT_FLOAT, None)
}

pub fn placeholder_i32(name: &str) -> tfpb::node_def::NodeDef {
    placeholder(name, DataType::DT_INT32, None)
}
