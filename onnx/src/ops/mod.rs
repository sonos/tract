use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb;
use crate::pb::*;
use std::convert::TryInto;
use tract_core::internal::*;

mod array;
mod category_mapper;
mod logic;
mod math;
mod nn;
pub mod rec;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Cast", cast);
    reg.insert("Constant", konst);
    reg.insert("Identity", |_, _| {
        Ok((Box::new(::tract_core::ops::identity::Identity::default()), vec![]))
    });
    category_mapper::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    array::register_all_ops(reg);
    rec::register_all_ops(reg);
}

fn konst(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let v = node.get_attr("value")?;
    Ok((Box::new(::tract_core::ops::konst::Const::for_tensor(v)), vec![]))
}

fn cast(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    use protobuf::ProtobufEnum;
    let to = node.get_attr("to")?;
    let to = pb::TensorProto_DataType::from_i32(to)
        .ok_or_else(|| format!("Cannot convert integer {} into a TensorProto_DataType", to))?;
    Ok((Box::new(::tract_core::ops::cast::Cast::new(to.try_into()?)), vec![]))
}
