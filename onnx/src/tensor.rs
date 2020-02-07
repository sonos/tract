use crate::pb::tensor_proto::DataType;
use crate::pb::*;
use prost::Message;
use std::convert::{TryFrom, TryInto};
use tract_core::internal::*;
use tract_core::infer::*;
use tract_core::*;

impl TryFrom<DataType> for DatumType {
    type Error = TractError;
    fn try_from(t: DataType) -> TractResult<DatumType> {
        match t {
            DataType::Bool => Ok(DatumType::Bool),
            DataType::Uint8 => Ok(DatumType::U8),
            DataType::Uint16 => Ok(DatumType::U16),
            DataType::Int8 => Ok(DatumType::I8),
            DataType::Int16 => Ok(DatumType::I16),
            DataType::Int32 => Ok(DatumType::I32),
            DataType::Int64 => Ok(DatumType::I64),
            DataType::Float16 => Ok(DatumType::F16),
            DataType::Float => Ok(DatumType::F32),
            DataType::Double => Ok(DatumType::F64),
            DataType::String => Ok(DatumType::String),
            _ => Err(format!("Unknown DatumType {:?}", t))?,
        }
    }

}

impl<'a> TryFrom<&'a type_proto::Tensor> for InferenceFact {
    type Error = TractError;
    fn try_from(t: &'a type_proto::Tensor) -> TractResult<InferenceFact> {
        let mut fact = InferenceFact::default();
        fact = fact.with_datum_type(DataType::from_i32(t.elem_type).unwrap().try_into()?);
        if let Some(shape) = &t.shape {
            let shape: TVec<DimFact> = shape
                .dim
                .iter()
                .map(|d| {
                    let mut fact = DimFact::default();
                    if let Some(tensor_shape_proto::dimension::Value::DimValue(v)) = d.value {
                        if v > 0 {
                            fact = DimFact::from(v.to_dim())
                        }
                    }
                    fact
                })
                .collect();
            fact = fact.with_shape(ShapeFactoid::closed(shape));
        }
        Ok(fact)
    }
}

impl TryFrom<type_proto::Tensor> for InferenceFact {
    type Error = TractError;
    fn try_from(t: type_proto::Tensor) -> TractResult<InferenceFact> {
        (&t).try_into()
    }
}

impl<'a> TryFrom<&'a TensorProto> for Tensor {
    type Error = TractError;
    fn try_from(t: &TensorProto) -> TractResult<Tensor> {
        let dt = DataType::from_i32(t.data_type).unwrap().try_into()?;
        let shape: Vec<usize> = t.dims.iter().map(|&i| i as usize).collect();
        if t.raw_data.len() > 0 {
            unsafe {
                match dt {
                    DatumType::U8 => Tensor::from_raw::<u8>(&*shape, &*t.raw_data),
                    DatumType::U16 => Tensor::from_raw::<u16>(&*shape, &*t.raw_data),
                    DatumType::I8 => Tensor::from_raw::<i8>(&*shape, &*t.raw_data),
                    DatumType::I16 => Tensor::from_raw::<i16>(&*shape, &*t.raw_data),
                    DatumType::I32 => Tensor::from_raw::<i32>(&*shape, &*t.raw_data),
                    DatumType::I64 => Tensor::from_raw::<i64>(&*shape, &*t.raw_data),
                    DatumType::F16 => Tensor::from_raw::<f16>(&*shape, &*t.raw_data),
                    DatumType::F32 => Tensor::from_raw::<f32>(&*shape, &*t.raw_data),
                    DatumType::F64 => Tensor::from_raw::<f64>(&*shape, &*t.raw_data),
                    DatumType::Bool => Ok(Tensor::from_raw::<u8>(&*shape, &*t.raw_data)?
                        .into_array::<u8>()?
                        .mapv(|x| x != 0)
                        .into()),
                    _ => unimplemented!("FIXME, raw tensor loading"),
                }
            }
        } else {
            use ndarray::Array;
            let it = match dt {
                DatumType::Bool => {
                    Array::from_shape_vec(&*shape, t.int32_data.iter().map(|&x| x != 0).collect())?
                        .into()
                }
                DatumType::U8 => {
                    Array::from_shape_vec(&*shape, t.int32_data.iter().map(|&x| x as u8).collect())?
                        .into()
                }
                DatumType::U16 => Array::from_shape_vec(
                    &*shape,
                    t.int32_data.iter().map(|&x| x as u16).collect(),
                )?
                .into(),
                DatumType::I8 => {
                    Array::from_shape_vec(&*shape, t.int32_data.iter().map(|&x| x as i8).collect())?
                        .into()
                }
                DatumType::I16 => Array::from_shape_vec(
                    &*shape,
                    t.int32_data.iter().map(|&x| x as i16).collect(),
                )?
                .into(),
                DatumType::I32 => {
                    Array::from_shape_vec(&*shape, t.int32_data.to_vec())?.into()
                }
                DatumType::I64 => {
                    Array::from_shape_vec(&*shape, t.int64_data.to_vec())?.into()
                }
                DatumType::F32 => {
                    Array::from_shape_vec(&*shape, t.float_data.to_vec())?.into()
                }
                DatumType::F64 => {
                    Array::from_shape_vec(&*shape, t.double_data.to_vec())?.into()
                }
                DatumType::String => {
                    let strings = t
                        .string_data
                        .iter()
                        .cloned()
                        .map(String::from_utf8)
                        .collect::<Result<Vec<String>, _>>()
                        .map_err(|_| format!("Invalid UTF8 buffer"))?;
                    Array::from_shape_vec(&*shape, strings)?.into()
                }
                _ => unimplemented!("FIXME, struct tensor loading"),
            };
            Ok(it)
        }
    }
}

impl TryFrom<TensorProto> for Tensor {
    type Error = TractError;
    fn try_from(t: TensorProto) -> TractResult<Tensor> {
        (&t).try_into()
    }
}

pub fn proto_from_reader<R: ::std::io::Read>(mut r: R) -> TractResult<TensorProto> {
    let mut v = vec![];
    r.read_to_end(&mut v)?;
    let b = bytes::Bytes::from(v);
    TensorProto::decode(b).map_err(|e| format!("Can not parse protobuf input: {:?}", e).into())
}

pub fn from_reader<R: ::std::io::Read>(r: R) -> TractResult<Tensor> {
    proto_from_reader(r)?.try_into()
}
