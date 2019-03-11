use tract_core::ops as tractops;
use tract_core::ops::prelude::*;

use crate::tfpb::node_def::NodeDef;

pub fn register_all_ops(reg: &mut Framework<NodeDef>) {
    reg.insert("Abs", with_T!(tractops::math::Abs));
    reg.insert("Add", with_T!(tractops::math::Add::Bin));
    reg.insert("AddN", add_n);
    reg.insert("BiasAdd", with_T!(tractops::math::Add::Bin));
    reg.insert("Ceil", with_T!(tractops::math::Ceil));
    reg.insert("Div", with_T!(tractops::math::Div::Bin));
    reg.insert("FloorMod", with_T!(tractops::math::Rem::Bin));
    reg.insert("MatMul", mat_mul);
    reg.insert("Maximum", with_T!(tractops::math::Max::Bin));
    reg.insert("Minimum", with_T!(tractops::math::Min::Bin));
    reg.insert("Less", with_T!(tractops::logic::Lesser::Bin));
    reg.insert("Log", with_T!(tractops::math::Ln));
    reg.insert("Mul", with_T!(tractops::math::Mul::Bin));
    reg.insert("Pow", with_T!(tractops::math::Pow::Bin));
    reg.insert("Neg", with_T!(tractops::math::Neg));
    reg.insert("RealDiv", with_T!(tractops::math::Div::Bin));
    reg.insert("Rsqrt", with_T!(tractops::math::Rsqrt));
    reg.insert("Sub", with_T!(tractops::math::Sub::Bin));
    reg.insert("Tanh", with_T!(tractops::math::Tanh));
}

pub fn add_n(pb: &NodeDef) -> TractResult<Box<tractops::Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let n = pb.get_attr_int("N")?;
    Ok(Box::new(tractops::math::AddN::new(dtype.into(), Some(n))))
}

pub fn mat_mul(pb: &NodeDef) -> TractResult<Box<tractops::Op>> {
    let trans_a = pb.get_attr_bool("transpose_a")?;
    let trans_b = pb.get_attr_bool("transpose_b")?;
    Ok(Box::new(tract_core::ops::math::Gemm::new(
        1.0, 0.0, trans_a, trans_b, false,
    )))
}
