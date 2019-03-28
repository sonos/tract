mod slice;

use tract_core::ops as tractops;
use tract_core::ops::prelude::*;

use crate::model::OnnxOpRegister;
use crate::pb;
use crate::pb::NodeProto;
use num_traits::AsPrimitive;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Concat", concat);
    reg.insert("ConstantLike", constant_like);
    reg.insert("ConstantOfShape", constant_of_shape);
    reg.insert("Expand", |_| {
        Ok(Box::new(tractops::array::MultiBroadcastTo::default()))
    });
    reg.insert("EyeLike", eye_like);
    reg.insert("Flatten", flatten);
    reg.insert("Gather", gather);
    reg.insert("Pad", pad);
    reg.insert("Reshape", |_| {
        Ok(Box::new(tractops::array::Reshape::default()))
    });
    reg.insert("Shape", |_| {
        Ok(Box::new(tractops::array::Shape::new(DatumType::I64)))
    });
    reg.insert("Size", |_| {
        Ok(Box::new(tractops::array::Size::new(DatumType::I64)))
    });
    reg.insert("Transpose", transpose);
    reg.insert("Slice", slice);
    reg.insert("Split", split);
    reg.insert("Squeeze", squeeze);
    reg.insert("Unsqueeze", unsqueeze);
}

pub fn concat(node: &NodeProto) -> TractResult<Box<Op>> {
    let axis = node.get_attr("axis")?;
    Ok(Box::new(tractops::array::Concat::new(axis)))
}

pub fn make_const<T>(shape: &[usize], v: f32) -> TractResult<SharedTensor>
where
    T: Copy + Datum,
    f32: AsPrimitive<T>,
{
    Ok(::ndarray::Array::<T, _>::from_elem(shape, v.as_()).into())
}

pub fn constant_like(node: &NodeProto) -> TractResult<Box<Op>> {
    let value = node.get_attr_opt("value")?.unwrap_or(0.);
    if node.get_input().len() == 0 {
        use protobuf::ProtobufEnum;
        let dt = match node.get_attr_opt("dtype")? {
            Some(dt) => pb::TensorProto_DataType::from_i32(dt)
                .ok_or_else(|| {
                    format!("Can not convert integer {} into a TensorProto_DataType", dt)
                })?
                .tractify()?,
            None => f32::datum_type(),
        };
        let shape: Vec<usize> = node.get_attr_vec("shape")?;
        let tensor = dispatch_numbers!(self::make_const(dt)(&shape, value))?;
        Ok(Box::new(tractops::konst::Const::new(tensor)))
    } else {
        Ok(Box::new(tractops::array::ConstantLike::new(value)))
    }
}

pub fn constant_of_shape(node: &NodeProto) -> TractResult<Box<Op>> {
    let value = match node.get_attr_opt::<Tensor>("value")? {
        Some(val) => val.into_tensor(),
        None => make_const::<f32>(&vec![1], 0.0 as f32)?,
    };
    Ok(Box::new(tractops::array::ConstantOfShape::new(value)))
}

pub fn eye_like(node: &NodeProto) -> TractResult<Box<Op>> {
    use protobuf::ProtobufEnum;
    let dt = match node.get_attr_opt("dtype")? {
        Some(dt) => Some(
            pb::TensorProto_DataType::from_i32(dt)
                .ok_or_else(|| {
                    format!("Can not convert integer {} into a TensorProto_DataType", dt)
                })?
                .tractify()?,
        ),
        None => None,
    };
    let k = node.get_attr_opt("k")?.unwrap_or(0);
    Ok(Box::new(tractops::array::EyeLike::new(dt, k)))
}

pub fn flatten(node: &NodeProto) -> TractResult<Box<Op>> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok(Box::new(tractops::array::Flatten::new(axis)))
}

pub fn gather(node: &NodeProto) -> TractResult<Box<Op>> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    Ok(Box::new(tractops::array::Gather::new(axis)))
}

pub fn pad(node: &NodeProto) -> TractResult<Box<Op>> {
    let value = node.get_attr_opt("value")?;
    let mode = match node.get_attr_opt("mode")? {
        None | Some("constant") => None,
        Some(mode) => node.check_value("mode", match mode {
            "reflect" => Ok(Some(tractops::array::PadMode::Reflect)),
            "edge" => Ok(Some(tractops::array::PadMode::Edge)),
            _ => Err(mode)
        })?
    }.unwrap_or_else(||
        tractops::array::PadMode::Constant(value.unwrap_or(0.))
    );
    let pads = node.get_attr_tvec("pads")?;
    let rank = pads.len() / 2;
    let pads = (0..rank)
        .map(|ax| (pads[ax], pads[ax + rank]))
        .collect();
    Ok(Box::new(tractops::array::Pad::new(pads, mode)))
}

pub fn slice(node: &NodeProto) -> TractResult<Box<Op>> {
    let axes = node.get_attr_opt_vec("axes")?;
    let begin = node.get_attr_vec("starts")?;
    let end = node.get_attr_vec("ends")?;
    Ok(Box::new(slice::Slice::new(axes, begin, end)))
}

pub fn split(node: &NodeProto) -> TractResult<Box<Op>> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    let split = node.get_attr_opt_vec("split")?;
    Ok(Box::new(tractops::array::Split::new(axis, node.get_output().len(), split)))
}

pub fn squeeze(node: &NodeProto) -> TractResult<Box<Op>> {
    let axes = node.get_attr_opt_vec("axes")?;
    Ok(Box::new(tractops::array::Squeeze::new(axes)))
}

pub fn transpose(node: &NodeProto) -> TractResult<Box<Op>> {
    let perm = node.get_attr_opt_vec("perm")?;
    Ok(Box::new(tractops::array::PermuteAxes::new(perm)))
}

pub fn unsqueeze(node: &NodeProto) -> TractResult<Box<Op>> {
    let axes = node.get_attr_vec("axes")?;
    Ok(Box::new(tractops::array::AddDims::new(axes)))
}
