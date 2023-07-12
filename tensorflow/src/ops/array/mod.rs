use crate::model::TfOpRegister;
use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_core::ops::array::StridedSlice;

mod concatv2;
mod expand_dims;
mod fill;
mod gather_nd;
mod gather_v2;
mod pack;
mod pad;
mod squeeze;
mod transpose;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("ConcatV2", concatv2::build);
    reg.insert("ExpandDims", expand_dims::build);
    reg.insert("Fill", fill::fill);
    reg.insert("GatherNd", gather_nd::gather_nd);
    reg.insert("GatherV2", gather_v2::gather_v2);
    reg.insert("Pack", pack::pack);
    reg.insert("Pad", pad::pad);
    reg.insert("Range", |_, _| Ok(expand(tract_hir::ops::array::Range)));
    reg.insert("Reshape", |_, _| Ok(expand(tract_hir::ops::array::Reshape::new())));
    reg.insert("Shape", |_, _| Ok(expand(tract_hir::ops::array::Shape::new(DatumType::TDim))));
    reg.insert("Slice", slice);
    reg.insert("Squeeze", squeeze::squeeze);
    reg.insert("StridedSlice", strided_slice);
    reg.insert("Tile", |_, _| Ok(expand(::tract_hir::ops::array::Tile)));
    reg.insert("Transpose", transpose::transpose);
}

fn strided_slice(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let begin_mask = pb.get_attr_opt_int("begin_mask")?.unwrap_or(0);
    let end_mask = pb.get_attr_opt_int("end_mask")?.unwrap_or(0);
    let shrink_axis_mask = pb.get_attr_opt_int("shrink_axis_mask")?.unwrap_or(0);
    Ok(Box::new(StridedSlice {
        begin_mask,
        end_mask,
        shrink_axis_mask,
        optional_axes_input: None,
        optional_steps_input: Some(3),
    }))
}

fn slice(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(Box::new(StridedSlice {
        optional_axes_input: None,
        optional_steps_input: None,
        begin_mask: 0,
        end_mask: 0,
        shrink_axis_mask: 0,
    }))
}
