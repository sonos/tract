mod compress;
mod nonzero;
mod one_hot;
mod pad;
mod shape;
mod slice;
mod split;
mod squeeze;
mod topk;
mod trilu;
mod unsqueeze;

use tract_hir::internal::*;
use tract_hir::ops::array;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("ArrayFeatureExtractor", array_feature_extractor);
    reg.insert("Compress", compress::compress);
    reg.insert("Concat", concat);
    reg.insert("ConstantLike", constant_like);
    reg.insert("ConstantOfShape", constant_of_shape);
    reg.insert("Expand", |_, _| Ok((expand(array::MultiBroadcastTo), vec![])));
    reg.insert("EyeLike", eye_like);
    reg.insert("Flatten", flatten);
    reg.insert("Gather", gather);
    reg.insert("GatherElements", gather_elements);
    reg.insert("GatherND", gather_nd);
    reg.insert("NonZero", nonzero::non_zero);
    reg.insert("OneHot", one_hot::one_hot);
    reg.insert("Range", |_, _| Ok((expand(array::Range), vec![])));
    reg.insert("Pad", pad::pad);
    reg.insert("Reshape", |_, _| Ok((expand(array::Reshape::default()), vec![])));
    reg.insert("Scatter", scatter_elements);
    reg.insert("ScatterElements", scatter_elements);
    reg.insert("ScatterND", |_, _| Ok((Box::new(array::ScatterNd), vec![])));
    reg.insert("Shape", shape::shape);
    reg.insert("Size", |_, _| Ok((expand(array::Size::new(DatumType::TDim)), vec![])));
    reg.insert("Slice", slice::slice);
    reg.insert("Split", split::split);
    reg.insert("Squeeze", squeeze::squeeze);
    reg.insert("Tile", |_, _| Ok((expand(array::Tile), vec![])));
    reg.insert("TopK", topk::topk);
    reg.insert("Transpose", transpose);
    reg.insert("Trilu", trilu::trilu);
    reg.insert("Unsqueeze", unsqueeze::unsqueeze);
}

pub fn array_feature_extractor(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(array::ArrayFeatureExtractor), vec![]))
}

pub fn concat(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr("axis")?;
    Ok((expand(array::Concat::new(axis)), vec![]))
}

pub fn constant_like(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let value = node.get_attr_opt("value")?.unwrap_or(0.);
    if node.input.len() == 0 {
        let dt = node.get_attr_opt("dtype")?.unwrap_or(DatumType::F32);
        let shape: Vec<usize> = node.get_attr_vec("shape")?;
        let tensor =
            tensor0(value).cast_to_dt(dt)?.broadcast_scalar_to_shape(&shape)?.into_arc_tensor();
        Ok((Box::new(tract_hir::ops::konst::Const::new(tensor)?), vec![]))
    } else {
        Ok((Box::new(array::ConstantLike::new(value)), vec![]))
    }
}

pub fn constant_of_shape(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut value = match node.get_attr_opt("value")? {
        Some(val) => ctx.load_tensor(val)?.into_arc_tensor(),
        None => rctensor0(0.0),
    };
    if value.rank() > 0 {
        if value.len() != 1 {
            bail!("Expected scalar (or vector of length 1), got {:?}", value);
        }
        value = value.nth(0)?.into_arc_tensor();
    }
    Ok((expand(array::ConstantOfShape::new(value)), vec![]))
}

pub fn eye_like(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let dt = node.get_attr_opt("dtype")?;
    let k = node.get_attr_opt("k")?.unwrap_or(0);
    Ok((Box::new(array::EyeLike::new(dt, k)), vec![]))
}

pub fn flatten(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis: i64 = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((expand(array::Flatten::new(axis)), vec![]))
}

pub fn gather(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    Ok((expand(array::Gather::new(axis)), vec![]))
}

pub fn gather_elements(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    Ok((expand(array::GatherElements::new(axis)), vec![]))
}

pub fn gather_nd(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let batch_dims = node.get_attr_opt("batch_dims")?.unwrap_or(0);
    Ok((Box::new(array::GatherNd::new(batch_dims)), vec![]))
}

pub fn scatter_elements(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    Ok((expand(array::ScatterElements::new(axis)), vec![]))
}

pub fn transpose(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let perm = node.get_attr_opt_vec("perm")?;
    Ok((expand(array::PermuteAxes::new(perm.map(|t| t.into()))), vec![]))
}
