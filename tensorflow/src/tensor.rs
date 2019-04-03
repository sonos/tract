use crate::tfpb::tensor::TensorProto;
use crate::tfpb::tensor_shape::{TensorShapeProto, TensorShapeProto_Dim};
use crate::tfpb::types::DataType;
use crate::ToSharedTensor;
use tract_core::{DatumType, Tensor, ToTract, TractResult, Tractify, TVec};

impl Tractify<DataType> for DatumType {
    fn tractify(t: &DataType) -> TractResult<DatumType> {
        match t {
            &DataType::DT_BOOL => Ok(DatumType::Bool),
            &DataType::DT_UINT8 => Ok(DatumType::U8),
            &DataType::DT_UINT16 => Ok(DatumType::U16),
            &DataType::DT_INT8 => Ok(DatumType::I8),
            &DataType::DT_INT16 => Ok(DatumType::I16),
            &DataType::DT_INT32 => Ok(DatumType::I32),
            &DataType::DT_INT64 => Ok(DatumType::I64),
            &DataType::DT_HALF => Ok(DatumType::F16),
            &DataType::DT_FLOAT => Ok(DatumType::F32),
            &DataType::DT_DOUBLE => Ok(DatumType::F64),
            &DataType::DT_STRING => Ok(DatumType::String),
            _ => Err(format!("Unknown DatumType {:?}", t))?,
        }
    }
}

impl Tractify<TensorShapeProto> for TVec<usize> {
    fn tractify(t: &TensorShapeProto) -> TractResult<TVec<usize>> {
        Ok(t.get_dim()
        .iter()
        .map(|d| d.size as usize)
        .collect::<TVec<_>>())
    }
}

impl ToSharedTensor<DataType> for DatumType {
    fn to_tf(&self) -> TractResult<DataType> {
        match self {
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

impl Tractify<TensorProto> for Tensor {
    fn tractify(t: &TensorProto) -> TractResult<Tensor> {
        let dtype = t.get_dtype();
        let dims:TVec<usize> = t.get_tensor_shape().tractify()?;
        let rank = dims.len();
        let content = t.get_tensor_content();
        let mat: Tensor = if content.len() != 0 {
            unsafe {
                match dtype {
                    DataType::DT_FLOAT => Self::from_raw::<f32>(&dims, content)?,
                    DataType::DT_INT32 => Self::from_raw::<i32>(&dims, content)?,
                    DataType::DT_INT64 => Self::from_raw::<i64>(&dims, content)?,
                    _ => unimplemented!("missing type {:?}", dtype),
                }
            }
        } else {
            use ndarray::Array;
            match dtype {
                DataType::DT_INT32 => {
                    Array::from_shape_vec(&*dims, t.get_int_val().to_vec())?.into()
                }
                DataType::DT_INT64 => {
                    Array::from_shape_vec(&*dims, t.get_int64_val().to_vec())?.into()
                }
                DataType::DT_FLOAT => {
                    Array::from_shape_vec(&*dims, t.get_float_val().to_vec())?.into()
                }
                _ => unimplemented!("missing type {:?}", dtype),
            }
        };
        assert_eq!(rank, mat.shape().len());
        Ok(mat)
    }
}

impl ToSharedTensor<TensorProto> for Tensor {
    fn to_tf(&self) -> TractResult<TensorProto> {
        let mut shape = TensorShapeProto::new();
        let dims = self
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
        match self.datum_type() {
            DatumType::F32 => {
                tensor.set_dtype(DatumType::F32.to_tf()?);
                tensor.set_float_val(self.to_array_view::<f32>()?.iter().cloned().collect());
            }
            DatumType::F64 => {
                tensor.set_dtype(DatumType::F64.to_tf()?);
                tensor.set_double_val(self.to_array_view::<f64>()?.iter().cloned().collect());
            }
            DatumType::I32 => {
                tensor.set_dtype(DatumType::I32.to_tf()?);
                tensor.set_int_val(self.to_array_view::<i32>()?.iter().cloned().collect());
            }
            DatumType::I64 => {
                tensor.set_dtype(DatumType::I64.to_tf()?);
                tensor.set_int64_val(self.to_array_view::<i64>()?.iter().cloned().collect());
            }
            _ => unimplemented!("missing type {:?}", self.datum_type()),
        }
        Ok(tensor)
    }
}
