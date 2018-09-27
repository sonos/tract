use tfdeploy::ops::prelude::*;
use tfdeploy::ops as tfdops;

use ops::OpRegister;
use pb::NodeProto;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Constant", constant);
    reg.insert("Expand", |_| {
        Ok(Box::new(tfdops::array::MultiBroadcastTo::default()))
    });
}

pub fn constant(node: &NodeProto) -> TfdResult<Box<Op>> {
    let value: Tensor = node.get_attr_tensor("value")?;
    Ok(Box::new(tfdops::konst::Const::new(value.into())))
}
