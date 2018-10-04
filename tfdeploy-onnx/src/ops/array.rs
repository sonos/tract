use tfdeploy::ops::prelude::*;
use tfdeploy::ops as tfdops;

use ops::OpRegister;
use pb::NodeProto;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Concat", concat);
    reg.insert("Expand", |_| {
        Ok(Box::new(tfdops::array::MultiBroadcastTo::default()))
    });
}

pub fn concat(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis = node.get_attr_int("axis")?;
    Ok(Box::new(tfdops::array::Concat::new(axis as usize)))
}
