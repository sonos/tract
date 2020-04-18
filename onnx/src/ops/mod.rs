use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;

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
    reg.insert("Identity", |_, _| Ok((Box::new(ops::identity::Identity::default()), vec![])));
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
    let v = node.get_attr::<Tensor>("value")?;
    Ok((Box::new(tract_hir::ops::konst::Const(v.into())), vec![]))
}

fn cast(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let to = node.get_attr::<DatumType>("to")?;
    Ok((Box::new(ops::cast(to)), vec![]))
}
