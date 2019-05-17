use std::convert::TryInto;
use crate::model::OnnxOpRegister;
use crate::pb;
use crate::pb::NodeProto;
use tract_core::internal::*;

mod array;
mod logic;
mod math;
mod nn;
pub mod rec;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Cast", cast);
    reg.insert("Constant", konst);
    reg.insert("Identity", |_| Ok(Box::new(::tract_core::ops::identity::Identity::default())));
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    array::register_all_ops(reg);
    rec::register_all_ops(reg);
}

fn konst(node: &NodeProto) -> TractResult<Box<InferenceOp>> {
    let v = node.get_attr("value")?;
    Ok(Box::new(::tract_core::ops::konst::Const::for_tensor(v)))
}

fn cast(node: &NodeProto) -> TractResult<Box<InferenceOp>> {
    use protobuf::ProtobufEnum;
    let to = node.get_attr("to")?;
    let to = pb::TensorProto_DataType::from_i32(to)
        .ok_or_else(|| format!("Cannot convert integer {} into a TensorProto_DataType", to))?;
    Ok(Box::new(::tract_core::ops::cast::Cast::new(to.try_into()?)))
}
