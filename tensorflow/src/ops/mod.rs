use std::collections::HashMap;

use tract_core::ops::Op;
use tract_core::TractResult;

use tfpb::node_def::NodeDef;

#[macro_use]
mod macros;

pub mod array;
pub mod math;
pub mod nn;

pub type OpRegister = HashMap<&'static str, fn(&NodeDef) -> TractResult<Box<Op>>>;

pub struct OpBuilder(OpRegister);

impl OpBuilder {
    pub fn new() -> OpBuilder {
        let mut reg = OpRegister::new();
        array::register_all_ops(&mut reg);
        math::register_all_ops(&mut reg);
        nn::register_all_ops(&mut reg);
        reg.insert("Const", konst);
        reg.insert("Placeholder", placeholder);
        OpBuilder(reg)
    }

    pub fn build(&self, pb: &NodeDef) -> TractResult<Box<Op>> {
        match self.0.get(pb.get_op()) {
            Some(builder) => builder(pb),
            None => Ok(Box::new(::tract_core::ops::unimpl::UnimplementedOp(
                pb.get_op().to_string(),
                format!("{:?}", pb),
            ))),
        }
    }
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
