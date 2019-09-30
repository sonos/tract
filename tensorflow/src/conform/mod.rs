#![allow(unused)]
#![allow(deprecated)]
#![allow(non_snake_case)]

error_chain! {
    foreign_links {
        Io(::std::io::Error);
        NdarrayShape(tract_core::ndarray::ShapeError);
        Protobuf(::protobuf::ProtobufError);
        StrUtf8(::std::str::Utf8Error);
    }
    errors {
        TFString {}
    }
}

impl ::std::convert::From<::tensorflow::Status> for Error {
    fn from(tfs: ::tensorflow::Status) -> Error {
        format!("Tensorflow error: {:?}", tfs).into()
    }
}

pub mod tf;

pub use protobuf::Message;

use crate::tfpb;
use crate::tfpb::tensor_shape::{TensorShapeProto, TensorShapeProto_Dim};
use crate::tfpb::types::DataType;
use tract_core::internal::*;

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
    let mut shape = TensorShapeProto::new();
    shape.set_dim(
        dims.iter()
            .map(|&d| {
                let mut dim = TensorShapeProto_Dim::new();
                dim.set_size(d as i64);
                dim
            })
            .collect(),
    );
    shape
}

pub fn const_f32(name: &str, t: &Tensor) -> tfpb::node_def::NodeDef {
    let mut tf = crate::tfpb::tensor_f32(
        t.shape().iter().cloned().collect(),
        t.to_array_view::<f32>().unwrap().iter().cloned().collect(),
    );
    tfpb::node().name(name).op("Const").attr("dtype", DataType::DT_FLOAT).attr("value", tf)
}

pub fn placeholder_f32(name: &str) -> tfpb::node_def::NodeDef {
    placeholder(name, DataType::DT_FLOAT, None)
}

pub fn const_i32(name: &str, t: &Tensor) -> tfpb::node_def::NodeDef {
    let mut tf = crate::tfpb::tensor_i32(
        t.shape().iter().cloned().collect(),
        t.to_array_view::<i32>().unwrap().iter().cloned().collect(),
    );
    tfpb::node().name(name).op("Const").attr("dtype", DataType::DT_INT32).attr("value", tf)
}

pub fn placeholder_i32(name: &str) -> tfpb::node_def::NodeDef {
    placeholder(name, DataType::DT_INT32, None)
}
