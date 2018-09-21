use tfdeploy::ops::prelude::*;
use pb::NodeProto;

pub type OpRegister = HashMap<&'static str, fn(&NodeProto) -> Result<Box<Op>>>;

pub struct OpBuilder(OpRegister);

impl OpBuilder {
    pub fn new() -> OpBuilder {
        let mut reg = OpRegister::new();
        reg.insert("Const", konst);
        /*
        array::register_all_ops(&mut reg);
        math::register_all_ops(&mut reg);
        nn::register_all_ops(&mut reg);
        reg.insert("Placeholder", ::ops::source::Source::build);
        */
        OpBuilder(reg)
    }

    pub fn build(&self, pb: &NodeProto) -> Result<Box<Op>> {
        match self.0.get(pb.get_op_type()) {
            Some(builder) => builder(pb),
            None => Ok(Box::new(::tfdeploy::ops::unimpl::UnimplementedOp(
                pb.get_op_type().to_string(),
                format!("{:?}", pb)
            ))),
        }
    }
}

fn konst(node: &NodeProto) -> Result<Box<Op>> {
    let v = node.get_attr_tensor("value")?;
    Ok(Box::new(::tfdeploy::ops::konst::Const::for_tensor(v)))
}
