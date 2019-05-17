use crate::tfpb::node_def::NodeDef;
use tract_core::internal::*;
use tract_core::ops::cnn::*;

pub fn avgpool(pb: &NodeDef) -> TractResult<Box<InferenceOp>> {
    let ksize: Vec<usize> = pb.get_attr_list_int("ksize")?;
    let data_format = super::data_format(pb)?;
    let kshape = data_format.shape(ksize);
    let strides = super::strides(pb)?;
    let padding = super::padding(pb)?;
    Ok(Box::new(AvgPool::new(
        PoolSpec::new(
            data_format,
            kshape.hw_dims().into(),
            padding,
            Some(strides[kshape.hw_axes()].into()),
        ),
        false,
    )))
}

pub fn maxpool(pb: &NodeDef) -> TractResult<Box<InferenceOp>> {
    let ksize: Vec<usize> = pb.get_attr_list_int("ksize")?;
    let data_format = super::data_format(pb)?;
    let kshape = data_format.shape(ksize);
    let strides = super::strides(pb)?;
    let padding = super::padding(pb)?;
    Ok(Box::new(MaxPool::new(
        PoolSpec::new(
            data_format,
            kshape.hw_dims().into(),
            padding,
            Some(strides[kshape.hw_axes()].into()),
        ),
        None,
    )))
}
