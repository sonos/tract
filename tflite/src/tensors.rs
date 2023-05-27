use crate::tflite_generated::{tflite::TensorType, TensorType as BufferTensorType};
#[cfg(feature = "complex")]
use num_complex::Complex;
use tract_hir::internal::*;

impl TryFrom<BufferTensorType> for DatumType {
    type Error = TractError;
    fn try_from(t: BufferTensorType) -> TractResult<DatumType> {
        Ok(match t {
            BufferTensorType::FLOAT32 => DatumType::F32,
            BUfferTensorType::FLOAT16 => DatumType::F16,
            BufferTensorType::INT32 => DatumType::I32,
            BufferTensorType::UINT8 => DatumType::U8,
            BufferTensorType::INT64 => DatumType::I64,
            BufferTensorType::STRING => DatumType::String,
            BufferTensorType::BOOL => DatumType::Bool,
            BufferTensorType::INT16 => DatumType::I16,
            #[cfg(feature = "complex")]
            BufferTensorType::COMPLEX64 => DatumType::ComplexF64,
            TensorType::INT8 => DatumType::I8,
            TensorType::FLOAT64 => DatumType::F64,
            //TensorType::COMPLEX128 => DatumType::ComplexF64,
            TensorType::UINT64 => DatumType::U64,
            TensorType::RESOURCE => DatumType::Blob, //TODO: check this
            TensorType::VARIANT => DatumType::Blob,  //TODO: check this
            TensorType::UINT32 => DatumType::U32,
            TensorType::UINT16 => DatumType::U16,
            //TensorType::COMPLEX128 => DatumType::ComplexF64,
            //TensorType::UINT4 => {DatumType::U4},
            _ => bail!("Unknown DatumType {:?}", t),
        })
    }
}

fn create_tensor(shape: Vec<usize>, dt: DatumType, data: &[u8]) -> TractResult<Tensor> {
    unsafe {
        match dt {
            DatumType::U8 => Tensor::from_raw::<u8>(&shape, data),
            DatumType::U16 => Tensor::from_raw::<u16>(&shape, data),
            DatumType::U32 => Tensor::from_raw::<u32>(&shape, data),
            DatumType::U64 => Tensor::from_raw::<u64>(&shape, data),
            DatumType::I8 => Tensor::from_raw::<i8>(&shape, data),
            DatumType::I16 => Tensor::from_raw::<i16>(&shape, data),
            DatumType::I32 => Tensor::from_raw::<i32>(&shape, data),
            DatumType::I64 => Tensor::from_raw::<i64>(&shape, data),
            DatumType::F16 => Tensor::from_raw::<f16>(&shape, data),
            DatumType::F32 => Tensor::from_raw::<f32>(&shape, data),
            DatumType::F64 => Tensor::from_raw::<f64>(&shape, data),
            #[cfg(feature = "complex")]
            DatumType::ComplexF64 => Tensor::from_raw::<Complex<f64>>(&shape, data),
            DatumType::Bool => Ok(Tensor::from_raw::<u8>(&shape, data)?
                .into_array::<u8>()?
                .mapv(|x| x != 0)
                .into()),
            _ => unimplemented!("FIXME, raw tensor loading"),
        }
    }
}
