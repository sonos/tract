use tf::tfpb::node_def::NodeDef;
use tf::ops::OpRegister;
use ops::Op;
use Result;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Abs", with_T!(::ops::math::Abs));
    reg.insert("Add", with_T!(::ops::math::Add));
    reg.insert("AddN", add_n);
    reg.insert("BiasAdd", with_T!(::ops::math::Add));
    reg.insert("Div", with_T!(::ops::math::Div));
    reg.insert("FloorMod", with_T!(::ops::math::Rem));
    reg.insert("Mul", with_T!(::ops::math::Mul));
    reg.insert("Neg", with_T!(::ops::math::Neg));
    reg.insert("Rsqrt", with_T!(::ops::math::Rsqrt));
    reg.insert("Sub", with_T!(::ops::math::Sub));
    reg.insert("Tanh", with_T!(::ops::math::Tanh));
}

pub fn add_n(pb: &NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let n = pb.get_attr_int("N")?;
    Ok(Box::new(::ops::math::add_n::AddN::new(dtype, n)))
}

