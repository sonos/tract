mod compress;
mod slice;

use std::convert::TryInto;

use tract_core::internal::*;
use tract_core::ops as tractops;

use crate::model::{ OnnxOpRegister, ParsingContext };
use crate::pb;
use crate::pb::*;
use num_traits::AsPrimitive;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Compress", compress::compress);
    reg.insert("Concat", concat);
    reg.insert("ConstantLike", constant_like);
    reg.insert("ConstantOfShape", constant_of_shape);
    reg.insert("Expand", |_, _| Ok((Box::new(tractops::array::MultiBroadcastTo::default()), vec!())));
    reg.insert("EyeLike", eye_like);
    reg.insert("Flatten", flatten);
    reg.insert("Gather", gather);
    reg.insert("Pad", pad);
    reg.insert("Reshape", |_, _| Ok((Box::new(tractops::array::Reshape::default()), vec!())));
    reg.insert("Shape", |_, _| Ok((Box::new(tractops::array::Shape::new(DatumType::I64)), vec!())));
    reg.insert("Size", |_, _| Ok((Box::new(tractops::array::Size::new(DatumType::I64)), vec!())));
    reg.insert("Transpose", transpose);
    reg.insert("Tile", |_, _| Ok((Box::new(tractops::array::Tile::default()), vec!())));
    reg.insert("Slice", slice);
    reg.insert("Split", split);
    reg.insert("Squeeze", squeeze);
    reg.insert("Unsqueeze", unsqueeze);
}

pub fn concat(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let axis = node.get_attr("axis")?;
    Ok((Box::new(tractops::array::Concat::new(axis)), vec!()))
}

pub fn make_const<T>(shape: &[usize], v: f32) -> TractResult<Arc<Tensor>>
where
    T: Copy + Datum,
    f32: AsPrimitive<T>,
{
    Ok(::ndarray::Array::<T, _>::from_elem(shape, v.as_()).into_arc_tensor())
}

pub fn constant_like(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let value = node.get_attr_opt("value")?.unwrap_or(0.);
    if node.get_input().len() == 0 {
        use protobuf::ProtobufEnum;
        let dt = match node.get_attr_opt("dtype")? {
            Some(dt) => pb::TensorProto_DataType::from_i32(dt)
                .ok_or_else(|| {
                    format!("Can not convert integer {} into a TensorProto_DataType", dt)
                })?
                .try_into()?,
            None => f32::datum_type(),
        };
        let shape: Vec<usize> = node.get_attr_vec("shape")?;
        let tensor = dispatch_numbers!(self::make_const(dt)(&shape, value))?;
        Ok((Box::new(tractops::konst::Const::new(tensor)), vec!()))
    } else {
        Ok((Box::new(tractops::array::ConstantLike::new(value)), vec!()))
    }
}

pub fn constant_of_shape(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let value = match node.get_attr_opt::<Tensor>("value")? {
        Some(val) => val.into_arc_tensor(),
        None => make_const::<f32>(&vec![1], 0.0 as f32)?,
    };
    Ok((Box::new(tractops::array::ConstantOfShape::new(value)), vec!()))
}

pub fn eye_like(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    use protobuf::ProtobufEnum;
    let dt = match node.get_attr_opt("dtype")? {
        Some(dt) => Some(
            pb::TensorProto_DataType::from_i32(dt)
                .ok_or_else(|| {
                    format!("Can not convert integer {} into a TensorProto_DataType", dt)
                })?
                .try_into()?,
        ),
        None => None,
    };
    let k = node.get_attr_opt("k")?.unwrap_or(0);
    Ok((Box::new(tractops::array::EyeLike::new(dt, k)), vec!()))
}

pub fn flatten(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(tractops::array::Flatten::new(axis)), vec!()))
}

pub fn gather(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    Ok((Box::new(tractops::array::Gather::new(axis)), vec!()))
}

pub fn pad(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let value = node.get_attr_opt("value")?;
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
    .unwrap_or_else(|| tractops::array::PadMode::Constant(value.unwrap_or(0.)));
    let pads = node.get_attr_tvec("pads")?;
    let rank = pads.len() / 2;
    let pads = (0..rank).map(|ax| (pads[ax], pads[ax + rank])).collect();
    Ok((Box::new(tractops::array::Pad::new(pads, mode)), vec!()))
}

pub fn slice(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?;
    let begin = node.get_attr_vec("starts")?;
    let end = node.get_attr_vec("ends")?;
    Ok((Box::new(slice::Slice::new(axes, begin, end)), vec!()))
}

pub fn split(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    let split = node.get_attr_opt_vec("split")?;
    Ok((Box::new(tractops::array::Split::new(axis, node.get_output().len(), split)), vec!()))
}

pub fn squeeze(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?;
    Ok((Box::new(tractops::array::Squeeze::new(axes)), vec!()))
}

pub fn transpose(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let perm = node.get_attr_opt_vec("perm")?;
    Ok((Box::new(tractops::array::PermuteAxes::new(perm)), vec!()))
}

pub fn unsqueeze(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>,Vec<String>)> {
    let axes = node.get_attr_vec("axes")?;
    Ok((Box::new(tractops::array::AddDims::new(axes)), vec!()))
}
