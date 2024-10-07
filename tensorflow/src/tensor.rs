use tract_hir::internal::*;

use crate::tfpb::tensorflow::tensor_shape_proto::Dim;
use crate::tfpb::tensorflow::{TensorProto, TensorShapeProto};

use crate::tfpb::tensorflow::DataType;
use std::convert::TryFrom;

impl TryFrom<DataType> for DatumType {
    type Error = TractError;
    fn try_from(t: DataType) -> TractResult<DatumType> {
        match t {
            DataType::DtBool => Ok(DatumType::Bool),
            DataType::DtUint8 => Ok(DatumType::U8),
            DataType::DtUint16 => Ok(DatumType::U16),
            DataType::DtUint32 => Ok(DatumType::U32),
            DataType::DtUint64 => Ok(DatumType::U64),
            DataType::DtInt8 => Ok(DatumType::I8),
            DataType::DtInt16 => Ok(DatumType::I16),
            DataType::DtInt32 => Ok(DatumType::I32),
            DataType::DtInt64 => Ok(DatumType::I64),
            DataType::DtHalf => Ok(DatumType::F16),
            DataType::DtFloat => Ok(DatumType::F32),
            DataType::DtDouble => Ok(DatumType::F64),
            DataType::DtString => Ok(DatumType::Blob),
            _ => Err(format_err!("Unknown DatumType {:?}", t))?,
        }
    }
}

impl<'a> TryFrom<&'a TensorShapeProto> for TVec<isize> {
    type Error = TractError;
    fn try_from(t: &'a TensorShapeProto) -> TractResult<TVec<isize>> {
        Ok(t.dim.iter().map(|d| d.size as isize).collect::<TVec<_>>())
    }
}

impl<'a> TryFrom<&'a TensorShapeProto> for TVec<usize> {
    type Error = TractError;
    fn try_from(t: &'a TensorShapeProto) -> TractResult<TVec<usize>> {
        if t.dim.iter().any(|d| d.size < 0) {
            bail!("Negative dim found")
        }
        Ok(t.dim.iter().map(|d| d.size as usize).collect::<TVec<_>>())
    }
}

impl TryFrom<DatumType> for DataType {
    type Error = TractError;
    fn try_from(dt: DatumType) -> TractResult<DataType> {
        match dt {
            DatumType::Bool => Ok(DataType::DtBool),
            DatumType::U8 => Ok(DataType::DtUint8),
            DatumType::U16 => Ok(DataType::DtUint16),
            DatumType::U32 => Ok(DataType::DtUint32),
            DatumType::U64 => Ok(DataType::DtUint64),
            DatumType::I8 => Ok(DataType::DtInt8),
            DatumType::I16 => Ok(DataType::DtInt16),
            DatumType::I32 => Ok(DataType::DtInt32),
            DatumType::I64 => Ok(DataType::DtInt64),
            DatumType::F16 => Ok(DataType::DtHalf),
            DatumType::F32 => Ok(DataType::DtFloat),
            DatumType::F64 => Ok(DataType::DtDouble),
            DatumType::Blob => Ok(DataType::DtString),
            DatumType::String => Ok(DataType::DtString),
            DatumType::QI8(_) => Ok(DataType::DtQint8),
            DatumType::QU8(_) => Ok(DataType::DtQint8),
            DatumType::QI32(_) => Ok(DataType::DtQint32),
            _ => bail!("DatumType is not translatable in protobuf"),
        }
    }
}

fn tensor_from_repeated_field<T: Datum>(shape: &[usize], data: Vec<T>) -> TractResult<Tensor> {
    let t = if data.len() == 1 {
        tract_ndarray::ArrayD::from_elem(shape, data[0].clone()).into()
    } else {
        tract_ndarray::ArrayD::from_shape_vec(shape, data.to_vec())?.into()
    };
    Ok(t)
}

impl TryFrom<&TensorProto> for Tensor {
    type Error = TractError;
    fn try_from(t: &TensorProto) -> TractResult<Tensor> {
        let dims: TVec<usize> =
            t.tensor_shape.as_ref().unwrap().dim.iter().map(|x| x.size as _).collect();
        let rank = dims.len();
        let content = &t.tensor_content;
        let dtype = DataType::from_i32(t.dtype).unwrap();
        let mat: Tensor = if content.len() != 0 {
            unsafe {
                match dtype {
                    DataType::DtFloat => Self::from_raw::<f32>(&dims, content)?,
                    DataType::DtDouble => Self::from_raw::<f64>(&dims, content)?,
                    DataType::DtInt32 => Self::from_raw::<i32>(&dims, content)?,
                    DataType::DtInt64 => Self::from_raw::<i64>(&dims, content)?,
                    _ => unimplemented!("missing type (for get_tensor_content) {:?}", dtype),
                }
            }
        } else {
            match dtype {
                DataType::DtInt32 => tensor_from_repeated_field(&dims, t.int_val.to_vec())?,
                DataType::DtInt64 => tensor_from_repeated_field(&dims, t.int64_val.to_vec())?,
                DataType::DtFloat => tensor_from_repeated_field(&dims, t.float_val.to_vec())?,
                DataType::DtDouble => tensor_from_repeated_field(&dims, t.double_val.to_vec())?,
                DataType::DtString => {
                    let strings = t
                        .string_val
                        .iter()
                        .map(|s| Blob::try_from(&**s))
                        .collect::<TractResult<Vec<Blob>>>()?;
                    tensor_from_repeated_field(&dims, strings)?
                }
                _ => unimplemented!("missing type (for _val()) {:?}", t.dtype),
            }
        };
        assert_eq!(rank, mat.shape().len());
        Ok(mat)
    }
}

fn empty_tensor_proto() -> TensorProto {
    TensorProto {
        dtype: 0,
        tensor_shape: None,
        version_number: 0,
        tensor_content: vec![],
        half_val: vec![],
        float_val: vec![],
        double_val: vec![],
        int_val: vec![],
        string_val: vec![],
        scomplex_val: vec![],
        dcomplex_val: vec![],
        resource_handle_val: vec![],
        variant_val: vec![],
        uint32_val: vec![],
        uint64_val: vec![],
        int64_val: vec![],
        bool_val: vec![],
    }
}

impl TryFrom<&Tensor> for TensorProto {
    type Error = TractError;
    fn try_from(from: &Tensor) -> TractResult<TensorProto> {
        let mut tensor = empty_tensor_proto();
        let shape = TensorShapeProto {
            dim: from.shape().iter().map(|d| Dim { size: *d as _, name: String::new() }).collect(),
            unknown_rank: false,
        };
        tensor.tensor_shape = Some(shape);
        let dt = DataType::try_from(from.datum_type())?;
        tensor.dtype = dt.into();
        match from.datum_type() {
            DatumType::F32 => {
                tensor.float_val = from.to_array_view::<f32>()?.iter().cloned().collect();
            }
            DatumType::F64 => {
                tensor.double_val = from.to_array_view::<f64>()?.iter().cloned().collect();
            }
            DatumType::I32 => {
                tensor.int_val = from.to_array_view::<i32>()?.iter().cloned().collect();
            }
            DatumType::I64 => {
                tensor.int64_val = from.to_array_view::<i64>()?.iter().cloned().collect();
            }
            _ => unimplemented!("missing type {:?}", from.datum_type()),
        }
        Ok(tensor)
    }
}
