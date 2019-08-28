use tract_core::internal::*;
use tract_core::ops as tractops;

use crate::model::TfOpRegister;
use crate::tfpb::node_def::NodeDef;
use crate::model::ParsingContext;


mod max;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("Abs", with_T!(tractops::math::Abs));
    reg.insert("Add", |_,_| Ok(Box::new(tractops::math::add::bin())));
    reg.insert("AddN", add_n);
    reg.insert("BiasAdd", |_,_| Ok(Box::new(tractops::math::add::bin())));
    reg.insert("Ceil", with_T!(tractops::math::Ceil));
    reg.insert("Div", |_,_| Ok(Box::new(tractops::math::div::bin())));
    reg.insert("FloorMod", |_,_| Ok(Box::new(tractops::math::rem::bin())));
    reg.insert("MatMul", mat_mul);
    reg.insert("Max", max::max);
    reg.insert("Maximum", |_,_| Ok(Box::new(tractops::math::max::bin())));
    reg.insert("Minimum", |_,_| Ok(Box::new(tractops::math::min::bin())));
    reg.insert("Less", |_,_| Ok(Box::new(tractops::logic::lesser::bin())));
    reg.insert("Log", with_T!(tractops::math::Ln));
    reg.insert("Mul", |_,_| Ok(Box::new(tractops::math::mul::bin())));
    reg.insert("Pow", |_,_| Ok(Box::new(tractops::math::pow::bin())));
    reg.insert("Neg", with_T!(tractops::math::Neg));
    reg.insert("RealDiv", |_,_| Ok(Box::new(tractops::math::div::bin())));
    reg.insert("Rsqrt", with_T!(tractops::math::Rsqrt));
    reg.insert("Sub", |_,_| Ok(Box::new(tractops::math::sub::bin())));
    reg.insert("Tanh", with_T!(tractops::math::Tanh));
}

pub fn add_n(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let n = pb.get_attr_int("N")?;
    Ok(Box::new(tractops::math::AddN::new(dtype.into(), Some(n))))
}

pub fn mat_mul(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let trans_a = pb.get_attr_bool("transpose_a")?;
    let trans_b = pb.get_attr_bool("transpose_b")?;
    Ok(Box::new(tract_core::ops::math::Gemm::new(1.0, 0.0, trans_a, trans_b, false)))
}
