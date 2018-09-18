use *;
use onnx::Protobuf;
use onnx::pb::*;

impl Protobuf<TensorProto_DataType> for DatumType {
    fn from_pb(t: &TensorProto_DataType) -> Result<DatumType> {
        use self::TensorProto_DataType::*;
        match t {
            &UINT8 => Ok(DatumType::U8),
            &INT8 => Ok(DatumType::I8),
            &INT32 => Ok(DatumType::I32),
            &FLOAT => Ok(DatumType::F32),
            &DOUBLE => Ok(DatumType::F64),
            &STRING => Ok(DatumType::String),
            _ => Err(format!("Unknown DatumType {:?}", t))?,
        }
    }

    fn to_pb(&self) -> Result<TensorProto_DataType> {
        use self::TensorProto_DataType::*;
        match self {
            DatumType::U8 => Ok(UINT8),
            DatumType::I8 => Ok(INT8),
            DatumType::I32 => Ok(INT32),
            DatumType::F32 => Ok(FLOAT),
            DatumType::F64 => Ok(DOUBLE),
            DatumType::String => Ok(STRING),
            DatumType::TDim => bail!("Dimension is not translatable in protobuf"),
        }
    }
}

impl Protobuf<TypeProto_Tensor> for TensorFact {
    fn from_pb(t: &TypeProto_Tensor) -> Result<TensorFact> {
        let mut fact = TensorFact::default();
        if t.has_elem_type() {
            fact = fact.with_datum_type(DatumType::from_pb(&t.get_elem_type())?);
        }
        if t.has_shape() {
            let shape = t.get_shape();
            let shape:Vec<usize> = shape.get_dim().iter().map(|d| d.get_dim_value() as usize).collect();
            fact = fact.with_shape(shape)
        }
        Ok(fact)
    }

    fn to_pb(&self) -> Result<TypeProto_Tensor> {
        unimplemented!("FIXME");
    }
}
