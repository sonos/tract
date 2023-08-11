use crate::tflite::{Model, SubGraph};
use crate::tflite_generated::tflite::{TensorType, TensorType as BufferTensorType};
#[cfg(feature = "complex")]
use num_complex::Complex;
use tract_core::internal::*;
use tract_core::prelude::tract_itertools::Itertools;

impl TryFrom<BufferTensorType> for DatumType {
    type Error = TractError;
    fn try_from(t: BufferTensorType) -> TractResult<DatumType> {
        Ok(match t {
            BufferTensorType::FLOAT32 => DatumType::F32,
            BufferTensorType::FLOAT16 => DatumType::F16,
            BufferTensorType::INT32 => DatumType::I32,
            BufferTensorType::UINT8 => DatumType::U8,
            BufferTensorType::INT64 => DatumType::I64,
            BufferTensorType::STRING => DatumType::String,
            BufferTensorType::BOOL => DatumType::Bool,
            BufferTensorType::INT16 => DatumType::I16,
            #[cfg(feature = "complex")]
            BufferTensorType::COMPLEX64 => DatumType::ComplexF64, // TODO check this
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

impl TryFrom<DatumType> for BufferTensorType {
    type Error = TractError;
    fn try_from(value: DatumType) -> Result<Self, Self::Error> {
        Ok(match value.unquantized() {
            DatumType::Bool => BufferTensorType::BOOL,
            DatumType::U8 => BufferTensorType::UINT8,
            DatumType::U16 => BufferTensorType::UINT16,
            DatumType::U32 => BufferTensorType::UINT32,
            DatumType::U64 => BufferTensorType::UINT64,
            DatumType::I8 => BufferTensorType::INT8,
            DatumType::I16 => BufferTensorType::INT16,
            DatumType::I32 => BufferTensorType::INT32,
            DatumType::I64 => BufferTensorType::INT64,
            DatumType::F16 => BufferTensorType::FLOAT16,
            DatumType::F32 => BufferTensorType::FLOAT32,
            DatumType::F64 => BufferTensorType::FLOAT64,
            _ => bail!("Unsupported DatumType {:?}", value),
        })
    }
}

#[allow(dead_code)]
fn create_tensor(dt: DatumType, shape: &[usize], data: &[u8]) -> TractResult<Tensor> {
    unsafe {
        match dt {
            DatumType::U8 => Tensor::from_raw::<u8>(shape, data),
            DatumType::U16 => Tensor::from_raw::<u16>(shape, data),
            DatumType::U32 => Tensor::from_raw::<u32>(shape, data),
            DatumType::U64 => Tensor::from_raw::<u64>(shape, data),
            DatumType::I8 => Tensor::from_raw::<i8>(shape, data),
            DatumType::I16 => Tensor::from_raw::<i16>(shape, data),
            DatumType::I32 => Tensor::from_raw::<i32>(shape, data),
            DatumType::I64 => Tensor::from_raw::<i64>(shape, data),
            DatumType::F16 => Tensor::from_raw::<f16>(shape, data),
            DatumType::F32 => Tensor::from_raw::<f32>(shape, data),
            DatumType::F64 => Tensor::from_raw::<f64>(shape, data),
            #[cfg(feature = "complex")]
            DatumType::ComplexF64 => Tensor::from_raw::<Complex<f64>>(&shape, data), // TODO check this
            DatumType::Bool => {
                Ok(Tensor::from_raw::<u8>(shape, data)?.into_array::<u8>()?.mapv(|x| x != 0).into())
            }
            _ => unimplemented!("FIXME, raw tensor loading"),
        }
    }
}

pub fn flat_tensor_uses_per_axis_q<'m>(graph: &'m SubGraph<'m>, id: i32) -> bool {
    let flat = graph.tensors().unwrap().get(id as _);
    if let Some(qp) = flat.quantization() {
        if let (Some(scale), Some(zp)) = (qp.scale(), qp.zero_point()) {
            return !scale.iter().all_equal() || !zp.iter().all_equal();
        }
    }
    false
}

pub fn per_axis_q_params<'m>(
    graph: &'m SubGraph<'m>,
    id: i32,
) -> TractResult<(Vec<i32>, Vec<f32>)> {
    let flat = graph.tensors().unwrap().get(id as _);
    let Some(qp) = flat.quantization() else { bail!("Unquantized value") };
    let (Some(scale), Some(zp)) = (qp.scale(), qp.zero_point()) else { bail!("No ZP/scale found") };
    Ok((zp.iter().map(|i| i as i32).collect_vec(), scale.iter().collect_vec()))
}

pub fn flat_tensor_to_tract_fact<'m>(
    &model: &'m Model<'m>,
    graph: &'m SubGraph<'m>,
    id: i32,
) -> TractResult<(TypedFact, &'m str)> {
    let flat = graph.tensors().unwrap().get(id as _);
    let mut dt: DatumType = flat.type_().try_into()?;
    if let Some(qp) = flat.quantization() {
        if let (Some(scale), Some(zp)) = (qp.scale(), qp.zero_point()) {
            dt = dt.quantize(QParams::ZpScale { zero_point: zp.get(0) as _, scale: scale.get(0) })
        }
    }
    let mut fact = dt.fact(flat.shape().unwrap().iter().map(|d| d as usize).collect_vec());
    let buffer_ix = flat.buffer() as usize;
    if buffer_ix != 0 {
        let buffer = model.buffers().unwrap().get(flat.buffer() as usize);
        if let Some(data) = buffer.data() {
            let mut data = create_tensor(
                fact.datum_type.unquantized(),
                fact.shape.as_concrete().unwrap(),
                data.bytes(),
            )?;
            unsafe {
                data.set_datum_type(dt);
            };
            fact = data.into();
        }
    }
    Ok((fact, flat.name().unwrap()))
}

#[derive(Clone, Debug)]
pub struct PerAxisQ {
    axis: usize,
    zp: Vec<i32>,
    scale: Vec<f32>,
}
