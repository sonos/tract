use tract_core::ops::nn::*;
use tract_core::ops::prelude::*;
use crate::tfpb::node_def::NodeDef;

pub fn avgpool(pb: &NodeDef) -> TractResult<Box<Op>> {
    let ksize: Vec<usize> = pb.get_attr_list_int("ksize")?;
    let data_format = super::data_format(pb)?;
    let kshape = data_format.shape(ksize);
    let strides = super::strides(pb)?;
    let padding = super::padding(pb)?;
    Ok(Box::new(AvgPool::new(
        data_format,
        kshape.hw_dims().into(),
        padding,
        Some(strides[kshape.hw_axes()].into()),
        false,
    )))
}

pub fn maxpool(pb: &NodeDef) -> TractResult<Box<Op>> {
    let ksize: Vec<usize> = pb.get_attr_list_int("ksize")?;
    let data_format = super::data_format(pb)?;
    let kshape = data_format.shape(ksize);
    let strides = super::strides(pb)?;
    let padding = super::padding(pb)?;
    Ok(Box::new(MaxPool::new(
        data_format,
        kshape.hw_dims().into(),
        padding,
        Some(strides[kshape.hw_axes()].into()),
        None
    )))
}
