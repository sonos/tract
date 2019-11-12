use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_core::internal::*;
use tract_core::ops::array::StridedSlice;

pub fn build(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let begin_mask = pb.get_attr_opt_int("begin_mask")?.unwrap_or(0);
    let end_mask = pb.get_attr_opt_int("end_mask")?.unwrap_or(0);
    let shrink_axis_mask = pb.get_attr_opt_int("shrink_axis_mask")?.unwrap_or(0);
    Ok(Box::new(StridedSlice::tensorflow(begin_mask, end_mask, shrink_axis_mask)))
}
