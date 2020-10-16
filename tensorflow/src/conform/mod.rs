#![allow(unused)]
#![allow(deprecated)]
#![allow(non_snake_case)]

pub mod tf;

use crate::tfpb;
use crate::tfpb::tensorflow::tensor_shape_proto::Dim;
use crate::tfpb::tensorflow::{DataType, TensorProto, TensorShapeProto};
use std::convert::TryInto;
use tract_hir::internal::*;

pub fn placeholder<Shape: Into<Option<TensorShapeProto>>>(
    name: &str,
    t: DataType,
    shape: Shape,
) -> tfpb::tensorflow::NodeDef {
    let mut node = tfpb::node().name(name).op("Placeholder").attr("dtype", t);
    if let Some(shape) = shape.into() {
        node = node.attr("shape", shape)
    }
    node
}

pub fn tensor_shape(dims: &[usize]) -> TensorShapeProto {
    TensorShapeProto {
        dim: dims.iter().map(|&d| Dim { size: d as i64, name: String::new() }).collect(),
        unknown_rank: false,
    }
}

pub fn const_f32(name: &str, t: &Tensor) -> tfpb::tensorflow::NodeDef {
    let tf: TensorProto = t.cast_to::<f32>().unwrap().as_ref().try_into().unwrap();
    tfpb::node().name(name).op("Const").attr("dtype", DataType::DtFloat).attr("value", tf)
}

pub fn placeholder_f32(name: &str) -> tfpb::tensorflow::NodeDef {
    placeholder(name, DataType::DtFloat, None)
}

pub fn const_i32(name: &str, t: &Tensor) -> tfpb::tensorflow::NodeDef {
    let tf: TensorProto = t.cast_to::<i32>().unwrap().as_ref().try_into().unwrap();
    tfpb::node().name(name).op("Const").attr("dtype", DataType::DtInt32).attr("value", tf)
}

pub fn placeholder_i32(name: &str) -> tfpb::tensorflow::NodeDef {
    placeholder(name, DataType::DtInt32, None)
}
