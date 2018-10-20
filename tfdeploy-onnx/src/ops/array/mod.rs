mod slice;

use tfdeploy::ops as tfdops;
use tfdeploy::ops::prelude::*;

use ops::OpRegister;
use pb::NodeProto;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Concat", concat);
    reg.insert("Expand", |_| {
        Ok(Box::new(tfdops::array::MultiBroadcastTo::default()))
    });
    reg.insert("Flatten", flatten);
    reg.insert("Reshape", |_| {
        Ok(Box::new(tfdops::array::Reshape::default()))
    });
    reg.insert("Shape", |_| {
        Ok(Box::new(tfdops::array::Shape::default()))
    });
    reg.insert("Slice", slice);
    reg.insert("Squeeze", squeeze);
    reg.insert("Unsqueeze", unsqueeze);
}

pub fn concat(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis = node.get_attr_int("axis")?;
    Ok(Box::new(tfdops::array::Concat::new(axis as usize)))
}

pub fn flatten(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis = node.get_attr_opt_int("axis")?.unwrap_or(1);
    Ok(Box::new(tfdops::array::Flatten::new(axis as usize)))
}

pub fn slice(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axes = node.get_attr_opt_ints("axes")?;
    let begin = node.get_attr_ints("starts")?;
    let end = node.get_attr_ints("ends")?;
    Ok(Box::new(slice::Slice::new(
        axes.map(|a| a.into_iter().map(|&d| d as _).collect()),
        begin.iter().map(|&d| d as _).collect(),
        end.iter().map(|&d| d as _).collect(),
    )))
}

pub fn squeeze(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axes = node
        .get_attr_opt_ints("axes")?
        .map(|l| l.iter().map(|&a| a as usize).collect());
    Ok(Box::new(tfdops::array::Squeeze::new(axes)))
}

pub fn unsqueeze(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axes = node
        .get_attr_ints("axes")?
        .iter()
        .map(|&a| a as usize)
        .collect();
    Ok(Box::new(tfdops::array::AddDims::new(axes)))
}
