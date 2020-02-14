use tract_hir::internal::*;
use tract_hir::ops;

use crate::model::ParsingContext;
use crate::model::TfOpRegister;
use crate::tfpb::tensorflow::NodeDef;

mod reduce;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("Abs", |_, _| Ok(Box::new(ops::math::abs())));
    reg.insert("Add", |_, _| Ok(Box::new(ops::math::add::bin())));
    reg.insert("AddN", add_n);
    reg.insert("AddV2", |_, _| Ok(Box::new(ops::math::add::bin())));
    reg.insert("BiasAdd", |_, _| Ok(Box::new(ops::math::add::bin())));
    reg.insert("Ceil", |_, _| Ok(Box::new(ops::math::ceil())));
    reg.insert("Div", |_, _| Ok(Box::new(ops::math::div::bin())));
    reg.insert("FloorMod", |_, _| Ok(Box::new(ops::math::rem::bin())));
    reg.insert("MatMul", mat_mul);
    reg.insert("Max", reduce::max);
    reg.insert("Mean", reduce::mean);
    reg.insert("Min", reduce::min);
    reg.insert("Prod", reduce::prod);
    reg.insert("Sum", reduce::sum);
    reg.insert("Maximum", |_, _| Ok(Box::new(ops::math::max::bin())));
    reg.insert("Minimum", |_, _| Ok(Box::new(ops::math::min::bin())));
    reg.insert("Less", |_, _| Ok(Box::new(ops::logic::lesser::bin())));
    reg.insert("Log", |_, _| Ok(Box::new(ops::math::ln())));
    reg.insert("Mul", |_, _| Ok(Box::new(ops::math::mul::bin())));
    reg.insert("Pow", |_, _| Ok(Box::new(ops::math::pow::bin())));
    reg.insert("Neg", |_, _| Ok(Box::new(ops::math::neg())));
    reg.insert("RealDiv", |_, _| Ok(Box::new(ops::math::div::bin())));
    reg.insert("Rsqrt", |_, _| Ok(Box::new(ops::math::rsqrt())));
    reg.insert("Sub", |_, _| Ok(Box::new(ops::math::sub::bin())));
    reg.insert("Tanh", |_, _| Ok(Box::new(ops::math::tanh())));
}

pub fn add_n(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(Box::new(ops::binary::Nary(Box::new(ops::math::Add), false)))
}

pub fn mat_mul(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let trans_a = pb.get_attr_bool("transpose_a")?;
    let trans_b = pb.get_attr_bool("transpose_b")?;
    Ok(Box::new(ops::matmul::MatMul::default().with_a_trans(trans_a).with_b_trans(trans_b)))
}
