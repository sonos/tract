use tract_hir::internal::*;
use tract_hir::ops::cnn::PaddingSpec;
use tract_hir::ops::nn::{DataFormat, LayerSoftmax};

use crate::model::TfOpRegister;
use crate::tfpb::tensorflow::NodeDef;

pub mod conv2d;
pub mod dw_conv2d;
pub mod fused_batch_norm;
pub mod pools;
pub mod s2b;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("AvgPool", pools::avgpool);
    reg.insert("Conv2D", conv2d::conv2d);
    reg.insert("DepthwiseConv2dNative", dw_conv2d::depthwise_conv2d);
    reg.insert("FusedBatchNorm", fused_batch_norm::fused_batch_norm);
    reg.insert("MaxPool", pools::maxpool);
    reg.insert("Relu", |_, _| Ok(expand(tract_hir::ops::activations::Clip::new(Some(0.0), None))));
    reg.insert("Relu6", |_, _| {
        Ok(expand(tract_hir::ops::activations::Clip::new(Some(0.0), Some(6.0))))
    });
    reg.insert("Sigmoid", |_, _| Ok(tract_hir::ops::nn::sigmoid().into_hir()));
    reg.insert("Softmax", |_, _| Ok(expand(LayerSoftmax::new(1, true))));
    reg.insert("SpaceToBatchND", s2b::space_to_batch_nd);
    reg.insert("BatchToSpaceND", s2b::batch_to_space_nd);
}

pub fn strides(pb: &NodeDef) -> TractResult<Vec<usize>> {
    let strides: Vec<usize> = pb.get_attr_list_int("strides")?;
    if strides.len() != 4 || strides[0] != 1 && strides[3] != 1 {
        bail!("strides must be of the form [1, h, v, 1], found {:?}", strides)
    };
    Ok(strides)
}

pub fn data_format(pb: &NodeDef) -> TractResult<DataFormat> {
    let df = if pb.get_attr_opt_raw_str("data_format")?.unwrap_or(b"NHWC") == b"NHWC" {
        DataFormat::NHWC
    } else {
        DataFormat::NCHW
    };
    Ok(df)
}

pub fn padding(pb: &NodeDef) -> TractResult<PaddingSpec> {
    let padding = pb.get_attr_raw_str("padding")?;
    match padding {
        b"VALID" => Ok(PaddingSpec::Valid),
        b"SAME" => Ok(PaddingSpec::SameUpper),
        s => bail!("unsupported Padding {}", String::from_utf8_lossy(s)),
    }
}
