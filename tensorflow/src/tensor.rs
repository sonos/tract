use crate::tfpb::tensor::TensorProto;
use crate::tfpb::tensor_shape::{TensorShapeProto, TensorShapeProto_Dim};
use crate::tfpb::types::DataType;
use std::convert::{TryFrom, TryInto};
use tract_core::internal::*;

impl TryFrom<DataType> for DatumType {
    type Error = TractError;
    fn try_from(t: DataType) -> TractResult<DatumType> {
        match t {
            DataType::DT_BOOL => Ok(DatumType::Bool),
            DataType::DT_UINT8 => Ok(DatumType::U8),
            DataType::DT_UINT16 => Ok(DatumType::U16),
            DataType::DT_INT8 => Ok(DatumType::I8),
            DataType::DT_INT16 => Ok(DatumType::I16),
            DataType::DT_INT32 => Ok(DatumType::I32),
            DataType::DT_INT64 => Ok(DatumType::I64),
            DataType::DT_HALF => Ok(DatumType::F16),
            DataType::DT_FLOAT => Ok(DatumType::F32),
            DataType::DT_DOUBLE => Ok(DatumType::F64),
            DataType::DT_STRING => Ok(DatumType::String),
            _ => Err(format!("Unknown DatumType {:?}", t))?,
        }
    }
}

impl<'a> TryFrom<&'a TensorShapeProto> for TVec<isize> {
    type Error = TractError;
    fn try_from(t: &'a TensorShapeProto) -> TractResult<TVec<isize>> {
        Ok(t.get_dim().iter().map(|d| d.size as isize).collect::<TVec<_>>())
    }
}

impl<'a> TryFrom<&'a TensorShapeProto> for TVec<usize> {
    type Error = TractError;
    fn try_from(t: &'a TensorShapeProto) -> TractResult<TVec<usize>> {
        if t.get_dim().iter().any(|d| d.size<0) {
            bail!("Negative dim found")
        }
        Ok(t.get_dim().iter().map(|d| d.size as usize).collect::<TVec<_>>())
    }
}

impl TryFrom<DatumType> for DataType {
    type Error = TractError;
    fn try_from(dt: DatumType) -> TractResult<DataType> {
        match dt {
            DatumType::Bool => Ok(DataType::DT_BOOL),
            DatumType::U8 => Ok(DataType::DT_UINT8),
            DatumType::U16 => Ok(DataType::DT_UINT16),
            DatumType::I8 => Ok(DataType::DT_INT8),
            DatumType::I16 => Ok(DataType::DT_INT16),
            DatumType::I32 => Ok(DataType::DT_INT32),
            DatumType::I64 => Ok(DataType::DT_INT64),
            DatumType::F16 => Ok(DataType::DT_HALF),
            DatumType::F32 => Ok(DataType::DT_FLOAT),
            DatumType::F64 => Ok(DataType::DT_DOUBLE),
            DatumType::String => Ok(DataType::DT_STRING),
            DatumType::TDim => bail!("Dimension is not translatable in protobuf"),
        }
    }
}

fn tensor_from_repeated_field<T: Datum>(shape: &[usize], data: Vec<T>) -> TractResult<Tensor> {
    let t = if data.len() == 1 {
        tract_core::ndarray::ArrayD::from_elem(shape, data[0].clone()).into()
    } else {
        tract_core::ndarray::ArrayD::from_shape_vec(shape, data.to_vec())?.into()
    };
    Ok(t)
}

impl<'a> TryFrom<&'a TensorProto> for Tensor {
    type Error = TractError;
    fn try_from(t: &TensorProto) -> TractResult<Tensor> {
        let dtype = t.get_dtype();
        let dims: TVec<usize> = t.get_tensor_shape().try_into()?;
        let rank = dims.len();
        let content = t.get_tensor_content();
        let mat: Tensor = if content.len() != 0 {
            unsafe {
                match dtype {
                    DataType::DT_FLOAT => Self::from_raw::<f32>(&dims, content)?,
                    DataType::DT_DOUBLE => Self::from_raw::<f64>(&dims, content)?,
                    DataType::DT_INT32 => Self::from_raw::<i32>(&dims, content)?,
                    DataType::DT_INT64 => Self::from_raw::<i64>(&dims, content)?,
                    _ => unimplemented!("missing type (for get_tensor_content) {:?}", dtype),
                }
            }
        } else {
            match dtype {
                DataType::DT_INT32 => tensor_from_repeated_field(&*dims, t.get_int_val().to_vec())?,
                DataType::DT_INT64 => {
                    tensor_from_repeated_field(&*dims, t.get_int64_val().to_vec())?
                }
                DataType::DT_FLOAT => {
                    tensor_from_repeated_field(&*dims, t.get_float_val().to_vec())?
                }
                DataType::DT_DOUBLE => {
                    tensor_from_repeated_field(&*dims, t.get_double_val().to_vec())?
                }
                DataType::DT_STRING => {
                    let strings = t
                        .get_string_val()
                        .iter()
                        .map(|s| {
                            std::str::from_utf8(s).map(|s| s.to_owned()).map_err(|_| {
                                format!("Invalid UTF-8: {}", String::from_utf8_lossy(s)).into()
                            })
                        })
                        .collect::<TractResult<Vec<String>>>()?;
                    tensor_from_repeated_field(&*dims, strings)?
                }
                _ => unimplemented!("missing type (for _val()) {:?}", dtype),
            }
        };
        assert_eq!(rank, mat.shape().len());
        Ok(mat)
    }
}

impl<'a> TryFrom<&'a Tensor> for TensorProto {
    type Error = TractError;
    fn try_from(from: &Tensor) -> TractResult<TensorProto> {
        let mut shape = TensorShapeProto::new();
        let dims = from
            .shape()
            .iter()
            .map(|d| {
                let mut dim = TensorShapeProto_Dim::new();
                dim.size = *d as _;
                dim
            })
            .collect();
        shape.set_dim(::protobuf::RepeatedField::from_vec(dims));
        let mut tensor = TensorProto::new();
        tensor.set_tensor_shape(shape);
        match from.datum_type() {
            DatumType::F32 => {
                tensor.set_dtype(DatumType::F32.try_into()?);
                tensor.set_float_val(from.to_array_view::<f32>()?.iter().cloned().collect());
            }
            DatumType::F64 => {
                tensor.set_dtype(DatumType::F64.try_into()?);
                tensor.set_double_val(from.to_array_view::<f64>()?.iter().cloned().collect());
            }
            DatumType::I32 => {
                tensor.set_dtype(DatumType::I32.try_into()?);
                tensor.set_int_val(from.to_array_view::<i32>()?.iter().cloned().collect());
            }
            DatumType::I64 => {
                tensor.set_dtype(DatumType::I64.try_into()?);
                tensor.set_int64_val(from.to_array_view::<i64>()?.iter().cloned().collect());
            }
            _ => unimplemented!("missing type {:?}", from.datum_type()),
        }
        Ok(tensor)
    }
}
