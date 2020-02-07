use crate::model::TfOpRegister;
use tract_core::infer::*;
use tract_core::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

mod concatv2;
mod expand_dims;
mod fill;
mod gather;
mod gather_v2;
mod pack;
mod pad;
mod range;
mod slice;
mod squeeze;
mod transpose;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("ConcatV2", concatv2::build);
    reg.insert("ExpandDims", expand_dims::build);
    reg.insert("Fill", fill::fill);
    reg.insert("GatherNd", gather::gather_nd);
    reg.insert("GatherV2", gather_v2::gather_v2);
    reg.insert("Pack", pack::pack);
    reg.insert("Pad", pad::pad);
    reg.insert("Range", range::range);
    reg.insert("Reshape", |_, _| Ok(Box::new(::tract_core::ops::array::Reshape::new())));
    reg.insert("Shape", |_, _| Ok(Box::new(::tract_core::ops::array::Shape::new(DatumType::I32))));
    reg.insert("Slice", |_, _| Ok(Box::new(slice::Slice)));
    reg.insert("Squeeze", squeeze::squeeze);
    reg.insert("StridedSlice", strided_slice);
    reg.insert("Tile", |_, _| Ok(Box::new(::tract_core::ops::array::Tile)));
    reg.insert("Transpose", transpose::transpose);
}

pub fn strided_slice(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    use tract_core::ops::array::StridedSlice;
    let begin_mask = pb.get_attr_opt_int("begin_mask")?.unwrap_or(0);
    let end_mask = pb.get_attr_opt_int("end_mask")?.unwrap_or(0);
    let shrink_axis_mask = pb.get_attr_opt_int("shrink_axis_mask")?.unwrap_or(0);
    Ok(Box::new(StridedSlice::tensorflow(begin_mask, end_mask, shrink_axis_mask)))
}
