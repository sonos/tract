use tract_core::internal::*;
use tract_core::ops as tractops;
use tract_core::ops::cnn::{KernelFormat, PaddingSpec};
use tract_core::ops::nn::DataFormat;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use crate::pb_helpers::OptionExt;

use num_traits::AsPrimitive;
use tractops::nn::Reducer;

pub fn conv_integer(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let kernel_shape = node.get_attr_opt_tvec("kernel_shape")?;
    let group = node.get_attr_opt("group")?.unwrap_or(1);
    Ok((
        Box::new(tractops::cnn::Conv::new(
            DataFormat::NCHW,
            KernelFormat::OIHW,
            dilations(node)?,
            kernel_shape,
            pad(node)?,
            strides(node)?,
            group,
        )),
        vec![],
    ))
}
