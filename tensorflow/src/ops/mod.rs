use tract_core::ops::prelude::*;
use crate::tfpb::node_def::NodeDef;

#[macro_use]
mod macros;

pub mod array;
pub mod logic;
pub mod math;
pub mod nn;
pub mod quant;

pub fn op_register() -> OpRegister<NodeDef> {
    let mut reg = OpRegister::default();
    register_all_ops(&mut reg);
    reg
}

pub fn register_all_ops(reg: &mut OpRegister<NodeDef>) {
    array::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    quant::register_all_ops(reg);
    reg.insert("Const", konst);
    reg.insert("Placeholder", placeholder);
}


pub fn konst(node: &NodeDef) -> TractResult<Box<Op>> {
    let dtype = node.get_attr_datum_type("dtype")?;
    let mat = node.get_attr_tensor("value")?;

    if mat.datum_type() != dtype {
        bail!(
            "Const node {:?} doesn't have the expected {:?} type.",
            mat,
            dtype
        );
    }

    Ok(Box::new(::tract_core::ops::konst::Const::for_tensor(mat)))
}

pub fn placeholder(node: &NodeDef) -> TractResult<Box<Op>> {
    let dt = node.get_attr_datum_type("dtype")?;
    Ok(Box::new(::tract_core::ops::source::Source::new(
        ::tract_core::analyser::types::TensorFact::dt(dt.into()),
    )))
}
