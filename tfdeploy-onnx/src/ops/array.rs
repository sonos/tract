use tfdeploy::ops::prelude::*;
use tfdeploy::ops as tfdops;

use ops::OpRegister;
use pb::NodeProto;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Concat", concat);
    reg.insert("Expand", |_| {
        Ok(Box::new(tfdops::array::MultiBroadcastTo::default()))
    });
    reg.insert("Reshape", |_|
        Ok(Box::new(tfdops::array::Reshape::default())));
    reg.insert("Squeeze", squeeze);
    reg.insert("Unsqueeze", unsqueeze);
}

pub fn concat(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis = node.get_attr_int("axis")?;
    Ok(Box::new(tfdops::array::Concat::new(axis as usize)))
}

pub fn squeeze(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axes = node.get_attr_opt_ints("axes")?.map(|l| l.iter().map(|&a| a as usize).collect());
    Ok(Box::new(tfdops::array::Squeeze::new(axes)))
}

pub fn unsqueeze(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axes = node.get_attr_ints("axes")?.iter().map(|&a| a as usize).collect();
    Ok(Box::new(tfdops::array::Unsqueeze::new(axes)))
}
