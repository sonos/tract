use tf::tfpb::tensor::TensorProto;
use tf::tfpb::tensor_shape::{TensorShapeProto, TensorShapeProto_Dim};
use tf::tfpb::types::DataType;
use tf::ToTensorflow;
use TfdFrom;
use *;

impl TfdFrom<DataType> for DatumType {
    fn tfd_from(t: &DataType) -> Result<DatumType> {
        match t {
            &DataType::DT_UINT8 => Ok(DatumType::U8),
            &DataType::DT_INT8 => Ok(DatumType::I8),
            &DataType::DT_INT32 => Ok(DatumType::I32),
            &DataType::DT_FLOAT => Ok(DatumType::F32),
            &DataType::DT_DOUBLE => Ok(DatumType::F64),
            &DataType::DT_STRING => Ok(DatumType::String),
            _ => Err(format!("Unknown DatumType {:?}", t))?,
        }
    }
}

impl ToTensorflow<DataType> for DatumType {
    fn to_tf(&self) -> Result<DataType> {
        match self {
            DatumType::U8 => Ok(DataType::DT_UINT8),
            DatumType::I8 => Ok(DataType::DT_INT8),
            DatumType::I32 => Ok(DataType::DT_INT32),
            DatumType::F32 => Ok(DataType::DT_FLOAT),
            DatumType::F64 => Ok(DataType::DT_DOUBLE),
            DatumType::String => Ok(DataType::DT_STRING),
            DatumType::TDim => bail!("Dimension is not translatable in protobuf"),
        }
    }
}

impl TfdFrom<TensorProto> for Tensor {
    fn tfd_from(t: &TensorProto) -> Result<Tensor> {
        let dtype = t.get_dtype();
        let shape = t.get_tensor_shape();
        let dims = shape
            .get_dim()
            .iter()
            .map(|d| d.size as usize)
            .collect::<Vec<_>>();
        let rank = dims.len();
        let content = t.get_tensor_content();
        let mat: Tensor = if content.len() != 0 {
            match dtype {
                DataType::DT_FLOAT => Self::from_content::<f32, u8>(dims, content)?.into(),
                DataType::DT_INT32 => Self::from_content::<i32, u8>(dims, content)?.into(),
                _ => unimplemented!("missing type"),
            }
        } else {
            match dtype {
                DataType::DT_INT32 => Self::from_content::<i32, i32>(dims, t.get_int_val())?.into(),
                DataType::DT_FLOAT => {
                    Self::from_content::<f32, f32>(dims, t.get_float_val())?.into()
                }
                _ => unimplemented!("missing type"),
            }
        };
        assert_eq!(rank, mat.shape().len());
        Ok(mat)
    }
}

impl ToTensorflow<TensorProto> for Tensor {
    fn to_tf(&self) -> Result<TensorProto> {
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
        match self {
            &Tensor::F32(ref it) => {
                tensor.set_dtype(DatumType::F32.to_tf()?);
                tensor.set_float_val(it.iter().cloned().collect());
            }
            &Tensor::F64(ref it) => {
                tensor.set_dtype(DatumType::F64.to_tf()?);
                tensor.set_double_val(it.iter().cloned().collect());
            }
            &Tensor::I32(ref it) => {
                tensor.set_dtype(DatumType::I32.to_tf()?);
                tensor.set_int_val(it.iter().cloned().collect());
            }
            _ => unimplemented!("missing type"),
        }
        Ok(tensor)
    }
}
