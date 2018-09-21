use std::collections::HashMap;

use tfdeploy::ops::Op;
use tfdeploy::Result;

use tfpb::node_def::NodeDef;

#[macro_use]
mod macros;

pub mod array;
pub mod math;
pub mod nn;

pub type OpRegister = HashMap<&'static str, fn(&NodeDef) -> Result<Box<Op>>>;

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

    pub fn build(&self, pb: &NodeDef) -> Result<Box<Op>> {
        match self.0.get(pb.get_op()) {
            Some(builder) => builder(pb),
            None => Ok(Box::new(::tfdeploy::ops::unimpl::UnimplementedOp(
                pb.get_op().to_string(),
                format!("{:?}", pb)
            ))),
        }
    }
}

pub fn konst(node: &NodeDef) -> Result<Box<Op>> {
    let dtype = node.get_attr_datum_type("dtype")?;
    let mat = node.get_attr_tensor("value")?;

    if mat.datum_type() != dtype {
        bail!(
            "Const node {:?} doesn't have the expected {:?} type.",
            mat,
            dtype
        );
    }

    Ok(Box::new(::tfdeploy::ops::konst::Const::for_tensor(mat)))
}

pub fn placeholder(node: &NodeDef) -> Result<Box<Op>> {
    let dt = node.get_attr_datum_type("dtype")?;
    Ok(Box::new(::tfdeploy::ops::source::Source::new(
        ::tfdeploy::analyser::types::TensorFact::dt(dt.into()),
    )))
}
