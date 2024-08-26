use tract_hir::internal::*;
use tract_hir::ops;

use crate::model::ParsingContext;
use crate::model::TfOpRegister;
use crate::tfpb::tensorflow::NodeDef;

mod reduce;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("Abs", |_, _| Ok(ops::math::abs().into_hir()));
    reg.insert("Add", |_, _| Ok(ops::math::Add.into_hir()));
    reg.insert("AddN", add_n);
    reg.insert("AddV2", |_, _| Ok(ops::math::Add.into_hir()));
    reg.insert("BiasAdd", |_, _| Ok(ops::math::Add.into_hir()));
    reg.insert("Ceil", |_, _| Ok(ops::math::ceil().into_hir()));
    reg.insert("Div", |_, _| Ok(ops::math::Div.into_hir()));
    reg.insert("FloorMod", |_, _| Ok(ops::math::Rem.into_hir()));
    reg.insert("MatMul", mat_mul);
    reg.insert("Max", reduce::max);
    reg.insert("Mean", reduce::mean);
    reg.insert("Min", reduce::min);
    reg.insert("Prod", reduce::prod);
    reg.insert("Sum", reduce::sum);
    reg.insert("Maximum", |_, _| Ok(ops::math::Max.into_hir()));
    reg.insert("Minimum", |_, _| Ok(ops::math::Min.into_hir()));
    reg.insert("Log", |_, _| Ok(ops::math::ln().into_hir()));
    reg.insert("Mul", |_, _| Ok(ops::math::Mul.into_hir()));
    reg.insert("Pow", |_, _| Ok(ops::math::Pow.into_hir()));
    reg.insert("Neg", |_, _| Ok(ops::math::neg().into_hir()));
    reg.insert("RealDiv", |_, _| Ok(ops::math::Div.into_hir()));
    reg.insert("Rsqrt", |_, _| Ok(ops::math::rsqrt().into_hir()));
    reg.insert("Sub", |_, _| Ok(ops::math::Sub.into_hir()));
    reg.insert("Tanh", |_, _| Ok(ops::math::tanh().into_hir()));
}

pub fn add_n(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(Box::new(ops::binary::Nary(Box::new(ops::math::Add), false)))
}

pub fn mat_mul(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let trans_a = pb.get_attr_bool("transpose_a")?;
    let trans_b = pb.get_attr_bool("transpose_b")?;
    Ok(expand(ops::matmul::MatMulInference::default().with_a_trans(trans_a).with_b_trans(trans_b)))
}
