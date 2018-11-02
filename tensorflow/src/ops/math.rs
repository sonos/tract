use tract_core::ops as tractops;

use ops::OpRegister;
use tfpb::node_def::NodeDef;
use tract_core::TractResult;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Abs", with_T!(tractops::math::Abs));
    reg.insert("Add", with_T!(tractops::math::Add::Bin));
    reg.insert("AddN", add_n);
    reg.insert("BiasAdd", with_T!(tractops::math::Add::Bin));
    reg.insert("Div", with_T!(tractops::math::Div::Bin));
    reg.insert("FloorMod", with_T!(tractops::math::Rem::Bin));
    reg.insert("Mul", with_T!(tractops::math::Mul::Bin));
    reg.insert("Neg", with_T!(tractops::math::Neg));
    reg.insert("Rsqrt", with_T!(tractops::math::Rsqrt));
    reg.insert("Sub", with_T!(tractops::math::Sub::Bin));
    reg.insert("Tanh", with_T!(tractops::math::Tanh));
}

pub fn add_n(pb: &NodeDef) -> TractResult<Box<tractops::Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let n = pb.get_attr_int("N")?;
    Ok(Box::new(tractops::math::AddN::new(dtype.into(), Some(n))))
}
