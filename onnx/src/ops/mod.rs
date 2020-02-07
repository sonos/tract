use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_core::infer::*;
use tract_core::internal::*;

mod array;
mod category_mapper;
mod logic;
mod math;
mod nn;
mod quant;
pub mod rec;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Cast", cast);
    reg.insert("Constant", konst);
    reg.insert("Identity", |_, _| {
        Ok((Box::new(::tract_core::ops::identity::Identity::default()), vec![]))
    });
    array::register_all_ops(reg);
    category_mapper::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    quant::register_all_ops(reg);
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
    let to = node.get_attr::<DatumType>("to")?;
    Ok((Box::new(::tract_core::ops::cast::cast(to)), vec![]))
}
