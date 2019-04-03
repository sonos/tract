use tract_core::ops::nn::{DataFormat, LayerSoftmax, PaddingSpec};
use tract_core::ops::prelude::*;

use crate::model::TfOpRegister;
use crate::tfpb::node_def::NodeDef;

pub mod conv2d;
pub mod fused_batch_norm;
pub mod pools;
pub mod s2b;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("AvgPool", pools::avgpool);
    reg.insert("Conv2D", conv2d::conv2d);
    reg.insert("FusedBatchNorm", fused_batch_norm::fused_batch_norm);
    reg.insert("MaxPool", pools::maxpool);
    reg.insert("Relu", with_T!(::tract_core::ops::nn::Relu));
    reg.insert("Sigmoid", with_T!(::tract_core::ops::nn::Sigmoid));
    reg.insert("Softmax", |_|  Ok(Box::new(LayerSoftmax::new(1))));
    reg.insert("SpaceToBatchND", s2b::space_to_batch_nd);
    reg.insert("BatchToSpaceND", s2b::batch_to_space_nd);
}

pub fn strides(pb: &NodeDef) -> TractResult<Vec<usize>> {
    let strides: Vec<usize> = pb.get_attr_list_int("strides")?;
    if strides.len() != 4 || strides[0] != 1 && strides[3] != 1 {
        Err(format!(
            "strides must be of the form [1, h, v, 1], found {:?}",
            strides
        ))?
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
        s => Err(format!(
            "unsupported Padding {}",
            String::from_utf8_lossy(s)
        ))?,
    }
}

