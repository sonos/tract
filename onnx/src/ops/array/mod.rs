mod compress;
mod slice;

use tract_core::internal::*;
use tract_core::infer::*;
use tract_core::ndarray;
use tract_core::ops as tractops;
use tract_core::hir;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use num_traits::AsPrimitive;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Compress", compress::compress);
    reg.insert("Concat", concat);
    reg.insert("ConstantLike", constant_like);
    reg.insert("ConstantOfShape", constant_of_shape);
    reg.insert("Expand", |_, _| {
        Ok((Box::new(hir::array::MultiBroadcastTo::default()), vec![]))
    });
    reg.insert("EyeLike", eye_like);
    reg.insert("Flatten", flatten);
    reg.insert("Gather", gather);
    reg.insert("Pad", pad);
    reg.insert("Reshape", |_, _| Ok((Box::new(tractops::array::Reshape::default()), vec![])));
    reg.insert("Shape", |_, _| Ok((Box::new(hir::array::Shape::new(DatumType::I64)), vec![])));
    reg.insert("Size", |_, _| Ok((Box::new(hir::array::Size::new(DatumType::I64)), vec![])));
    reg.insert("Transpose", transpose);
    reg.insert("Tile", |_, _| Ok((Box::new(tractops::array::Tile::default()), vec![])));
    reg.insert("Slice", slice::slice);
    reg.insert("Split", split);
    reg.insert("Squeeze", squeeze);
    reg.insert("Unsqueeze", unsqueeze);
}

pub fn concat(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr("axis")?;
    Ok((Box::new(hir::array::Concat::new(axis)), vec![]))
}

pub fn make_const<T>(shape: &[usize], v: f32) -> TractResult<Arc<Tensor>>
where
    T: Copy + Datum,
    f32: AsPrimitive<T>,
{
    Ok(ndarray::Array::<T, _>::from_elem(shape, v.as_()).into_arc_tensor())
}

pub fn constant_like(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let value = node.get_attr_opt("value")?.unwrap_or(0.);
    if node.input.len() == 0 {
        let dt = node.get_attr_opt("dtype")?.unwrap_or(f32::datum_type());
        let shape: Vec<usize> = node.get_attr_vec("shape")?;
        let tensor = dispatch_numbers!(self::make_const(dt)(&shape, value))?;
        Ok((Box::new(tractops::konst::Const::new(tensor)), vec![]))
    } else {
        Ok((Box::new(hir::array::ConstantLike::new(value)), vec![]))
    }
}

pub fn constant_of_shape(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let value = match node.get_attr_opt::<Tensor>("value")? {
        Some(val) => val.into_arc_tensor(),
        None => make_const::<f32>(&vec![1], 0.0 as f32)?,
    };
    Ok((Box::new(tractops::array::ConstantOfShape::new(value)), vec![]))
}

pub fn eye_like(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let dt = node.get_attr_opt("dtype")?;
    let k = node.get_attr_opt("k")?.unwrap_or(0);
    Ok((Box::new(hir::array::EyeLike::new(dt, k)), vec![]))
}

pub fn flatten(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(tractops::array::Flatten::new(axis)), vec![]))
}

pub fn gather(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    Ok((Box::new(tractops::array::Gather::new(axis)), vec![]))
}

pub fn pad(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let value: f32 = node.get_attr_opt("value")?.unwrap_or(0.0);
    let mode = match node.get_attr_opt("mode")? {
        None | Some("constant") => None,
        Some(mode) => node.check_value(
            "mode",
            match mode {
                "reflect" => Ok(Some(tractops::array::PadMode::Reflect)),
                "edge" => Ok(Some(tractops::array::PadMode::Edge)),
                _ => Err(mode),
            },
        )?,
    }
    .unwrap_or_else(|| tractops::array::PadMode::Constant(Arc::new(value.into())));
    let pads = node.get_attr_tvec("pads")?;
    let rank = pads.len() / 2;
    let pads = (0..rank).map(|ax| (pads[ax], pads[ax + rank])).collect();
    Ok((Box::new(tractops::array::Pad::new(pads, mode)), vec![]))
}

pub fn split(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    let split = node.get_attr_opt_vec("split")?;
    Ok((Box::new(hir::array::Split::new(axis, node.output.len(), split)), vec![]))
}

pub fn squeeze(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?;
    Ok((Box::new(hir::array::Squeeze::new(axes)), vec![]))
}

pub fn transpose(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let perm = node.get_attr_opt_vec("perm")?;
    Ok((Box::new(hir::array::PermuteAxes::new(perm)), vec![]))
}

pub fn unsqueeze(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_vec("axes")?;
    Ok((Box::new(hir::array::AddDims::new(axes)), vec![]))
}
