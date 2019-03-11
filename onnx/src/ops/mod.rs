use crate::Onnx;
use crate::pb;
use crate::pb::NodeProto;
use tract_core::ops::prelude::*;

mod array;
mod logic;
mod math;
mod nn;

pub fn register_all_ops(reg: &mut Onnx) {
    reg.insert("Cast", cast);
    reg.insert("Constant", konst);
    reg.insert("Identity", |_| {
        Ok(Box::new(::tract_core::ops::identity::Identity::default()))
    });
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    array::register_all_ops(reg);
}


fn konst(node: &NodeProto) -> TractResult<Box<Op>> {
    let v = node.get_attr("value")?;
    Ok(Box::new(::tract_core::ops::konst::Const::for_tensor(v)))
}

fn cast(node: &NodeProto) -> TractResult<Box<Op>> {
    use protobuf::ProtobufEnum;
    use tract_core::ToTract;
    let to = node.get_attr("to")?;
    let to = pb::TensorProto_DataType::from_i32(to)
        .ok_or_else(|| format!("Cannot convert integer {} into a TensorProto_DataType", to))?;
    Ok(Box::new(::tract_core::ops::cast::Cast::new(to.tractify()?)))
}
