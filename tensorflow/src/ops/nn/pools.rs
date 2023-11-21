use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_hir::internal::*;
use tract_hir::ops::cnn::*;

pub fn avgpool(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let ksize: Vec<usize> = pb.get_attr_list_int("ksize")?;
    let data_format = super::data_format(pb)?;
    let kshape = data_format.shape(ksize)?;
    let strides = super::strides(pb)?;
    let padding = super::padding(pb)?;
    Ok(expand(HirSumPool::new(
        PoolSpec::new(
            data_format,
            kshape.hw_dims().into(),
            padding,
            None,
            Some(strides[kshape.hw_axes()].into()),
            0,
            0,
        ),
        false,
        true,
    )))
}

pub fn maxpool(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let ksize: Vec<usize> = pb.get_attr_list_int("ksize")?;
    let data_format = super::data_format(pb)?;
    let kshape = data_format.shape(ksize)?;
    let strides = super::strides(pb)?;
    let padding = super::padding(pb)?;
    Ok(expand(HirMaxPool::new(
        PoolSpec::new(
            data_format,
            kshape.hw_dims().into(),
            padding,
            None,
            Some(strides[kshape.hw_axes()].into()),
            0,
            0,
        ),
        None,
    )))
}
