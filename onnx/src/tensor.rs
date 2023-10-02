use crate::model::{ParsingContext, TensorPlusPath};
use crate::pb::tensor_proto::DataType;
use crate::pb::*;
use prost::Message;
use std::convert::{TryFrom, TryInto};
use std::fs;
use std::path::{Path, PathBuf};
use tract_hir::internal::*;

impl TryFrom<DataType> for DatumType {
    type Error = TractError;
    fn try_from(t: DataType) -> TractResult<DatumType> {
        match t {
            DataType::Bool => Ok(DatumType::Bool),
            DataType::Uint8 => Ok(DatumType::U8),
            DataType::Uint16 => Ok(DatumType::U16),
            DataType::Uint32 => Ok(DatumType::U32),
            DataType::Uint64 => Ok(DatumType::U64),
            DataType::Int8 => Ok(DatumType::I8),
            DataType::Int16 => Ok(DatumType::I16),
            DataType::Int32 => Ok(DatumType::I32),
            DataType::Int64 => Ok(DatumType::I64),
            DataType::Float16 => Ok(DatumType::F16),
            DataType::Float => Ok(DatumType::F32),
            DataType::Double => Ok(DatumType::F64),
            DataType::String => Ok(DatumType::String),
            _ => bail!("Unknown DatumType {:?}", t),
        }
    }
}

pub fn translate_inference_fact(
    ctx: &ParsingContext,
    t: &type_proto::Tensor,
) -> TractResult<InferenceFact> {
    let mut fact = InferenceFact::default();
    fact = fact.with_datum_type(DataType::from_i32(t.elem_type).unwrap().try_into()?);
    if let Some(shape) = &t.shape {
        let shape: TVec<DimFact> = shape
            .dim
            .iter()
            .map(|d| match &d.value {
                Some(tensor_shape_proto::dimension::Value::DimValue(v)) if *v >= 0 => {
                    DimFact::from(v.to_dim())
                }
                Some(tensor_shape_proto::dimension::Value::DimParam(v)) => {
                    let sym = ctx.symbol_table.sym(v);
                    DimFact::from(sym.to_dim())
                }
                _ => DimFact::default(),
            })
            .collect();
        fact = fact.with_shape(ShapeFactoid::closed(shape));
    }
    Ok(fact)
}

#[cfg(target_family="wasm")]
fn extend_bytes_from_path(buf: &mut Vec<u8>, p: impl AsRef<Path>) -> TractResult<()> {
    use std::io::BufRead;

    let file = fs::File::open(p)?;
    let file_size = file.metadata()?.len() as usize;
    if buf.capacity() < file_size + buf.len() {
        buf.reserve(file_size);
    }

    let mut reader = std::io::BufReader::new(file);
    while reader.fill_buf()?.len() > 0 {
        buf.extend_from_slice(reader.buffer());
        reader.consume(reader.buffer().len());
    }
    Ok(())
}

#[cfg(all(any(windows, unix), not(target_os = "emscripten")))]
fn extend_bytes_from_path(buf: &mut Vec<u8>, p: impl AsRef<Path>) -> TractResult<()> {
    let file = fs::File::open(p)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    buf.extend_from_slice(&mmap);
    Ok(())
}

fn get_external_resources(t: &TensorProto, path: &str) -> TractResult<Vec<u8>> {
    let mut tensor_data: Vec<u8> = Vec::new();
    trace!("number of external file needed for this tensor: {}", t.external_data.len());
    for external_data in t.external_data.iter()
    // according to the onnx format, it is possible to have multiple files for one tensor
    {
        let p = PathBuf::from(format!("{}/{}", path, external_data.value));
        trace!("external file detected: {:?}", p);
        extend_bytes_from_path(&mut tensor_data, p)?;
        trace!("external file loaded");
    }
    Ok(tensor_data)
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
            DatumType::Bool => Ok(Tensor::from_raw::<u8>(&shape, data)?
                .into_array::<u8>()?
                .mapv(|x| x != 0)
                .into()),
            _ => unimplemented!("FIXME, raw tensor loading"),
        }
    }
}

fn common_tryfrom(t: &TensorProto, path: Option<&str>) -> TractResult<Tensor> {
    let dt = DataType::from_i32(t.data_type).unwrap().try_into()?;
    let shape: Vec<usize> = t.dims.iter().map(|&i| i as usize).collect();
    // detect if the tensor is rather in an external file than inside the onnx file directly
    let is_external = t.data_location.is_some() && t.data_location == Some(1);
    if t.raw_data.len() > 0 {
        create_tensor(shape, dt, &t.raw_data)
    } else if is_external {
        if let Some(model_path) = path {
            // external files will be loaded and fed to the tensor if necessary
            let external_data = get_external_resources(t, model_path)?;
            create_tensor(shape, dt, &external_data)
        } else {
            bail!("no model path was specified in the parsing context, yet external data was detected. aborting");
        }
    } else {
        use tract_ndarray::Array;
        let it = match dt {
            DatumType::Bool => {
                Array::from_shape_vec(&*shape, t.int32_data.iter().map(|&x| x != 0).collect())?
                    .into()
            }
            DatumType::U8 => {
                Array::from_shape_vec(&*shape, t.int32_data.iter().map(|&x| x as u8).collect())?
                    .into()
            }
            DatumType::U16 => {
                Array::from_shape_vec(&*shape, t.int32_data.iter().map(|&x| x as u16).collect())?
                    .into()
            }
            DatumType::U32 => Array::from_shape_vec(&*shape, t.int32_data.to_vec())?.into(),
            DatumType::U64 => Array::from_shape_vec(&*shape, t.int64_data.to_vec())?.into(),
            DatumType::I8 => {
                Array::from_shape_vec(&*shape, t.int32_data.iter().map(|&x| x as i8).collect())?
                    .into()
            }
            DatumType::I16 => {
                Array::from_shape_vec(&*shape, t.int32_data.iter().map(|&x| x as i16).collect())?
                    .into()
            }
            DatumType::I32 => Array::from_shape_vec(&*shape, t.int32_data.to_vec())?.into(),
            DatumType::I64 => Array::from_shape_vec(&*shape, t.int64_data.to_vec())?.into(),
            DatumType::F32 => Array::from_shape_vec(&*shape, t.float_data.to_vec())?.into(),
            DatumType::F64 => Array::from_shape_vec(&*shape, t.double_data.to_vec())?.into(),
            DatumType::String => {
                let strings = t
                    .string_data
                    .iter()
                    .cloned()
                    .map(String::from_utf8)
                    .collect::<Result<Vec<String>, _>>()
                    .context("Invalid UTF8 buffer")?;
                Array::from_shape_vec(&*shape, strings)?.into()
            }
            _ => unimplemented!("FIXME, struct tensor loading"),
        };
        Ok(it)
    }
}

impl TryFrom<TensorPlusPath<'_>> for Tensor {
    type Error = TractError;
    fn try_from(st: TensorPlusPath) -> TractResult<Tensor> {
        common_tryfrom(st.tensor, Some(st.model_path))
    }
}

impl<'a> TryFrom<&'a TensorProto> for Tensor {
    type Error = TractError;
    fn try_from(t: &TensorProto) -> TractResult<Tensor> {
        common_tryfrom(t, None)
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
    TensorProto::decode(b).context("Can not parse protobuf input")
}

pub fn from_reader<R: ::std::io::Read>(r: R) -> TractResult<Tensor> {
    proto_from_reader(r)?.try_into()
}
