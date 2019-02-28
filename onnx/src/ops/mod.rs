use crate::pb;
use crate::pb::NodeProto;
use tract_core::ops::prelude::*;

mod array;
mod logic;
mod math;
mod nn;

pub type OpRegister = HashMap<&'static str, fn(&NodeProto) -> TractResult<Box<Op>>>;

pub struct OpBuilder(OpRegister);

impl OpBuilder {
    pub fn new() -> OpBuilder {
        let mut reg = OpRegister::new();
        reg.insert("Cast", cast);
        reg.insert("Constant", konst);
        reg.insert("Identity", |_| {
            Ok(Box::new(::tract_core::ops::identity::Identity::default()))
        });
        logic::register_all_ops(&mut reg);
        math::register_all_ops(&mut reg);
        nn::register_all_ops(&mut reg);
        array::register_all_ops(&mut reg);
        OpBuilder(reg)
    }

    pub fn build(&self, pb: &NodeProto) -> TractResult<Box<Op>> {
        match self.0.get(pb.get_op_type()) {
            Some(builder) => builder(pb),
            None => Ok(Box::new(::tract_core::ops::unimpl::UnimplementedOp::new(
                pb.get_op_type(),
                format!("{:?}", pb),
            ))),
        }
    }
}

fn konst(node: &NodeProto) -> TractResult<Box<Op>> {
    let v = node.get_attr("value")?;
    Ok(Box::new(::tract_core::ops::konst::Const::for_tensor(v)))
}

fn cast(node: &NodeProto) -> TractResult<Box<Op>> {
    use protobuf::ProtobufEnum;
    use tract_core::ToTract;
    let to = node.get_attr_int("to")?;
    let to = pb::TensorProto_DataType::from_i32(to as i32)
        .ok_or_else(|| format!("Can not convert integer {} into a TensorProto_DataType", to))?;
    Ok(Box::new(::tract_core::ops::cast::Cast::new(to.tractify()?)))
}
