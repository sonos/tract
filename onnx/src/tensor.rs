use crate::data_resolver::ModelDataResolver;
use crate::model::ParsingContext;
use crate::pb::tensor_proto::DataType;
use crate::pb::*;
use prost::Message;
use std::convert::{TryFrom, TryInto};
use std::path::PathBuf;
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
    include_unknown_symbols: bool,
) -> TractResult<InferenceFact> {
    let mut fact = InferenceFact::default();
    fact = fact.with_datum_type(DataType::from_i32(t.elem_type).unwrap().try_into()?);
    if let Some(shape) = &t.shape {
        let shape: TVec<DimFact> = shape
            .dim
            .iter()
            .map(|d| -> TractResult<DimFact> {
                match &d.value {
                    Some(tensor_shape_proto::dimension::Value::DimValue(v)) if *v >= 0 => {
                        Ok(DimFact::from(v.to_dim()))
                    }
                    Some(tensor_shape_proto::dimension::Value::DimParam(v)) => {
                        if v == "?" || (v.starts_with("unk__") && !include_unknown_symbols) {
                            Ok(DimFact::default())
                        } else {
                            let dim = parse_tdim(&ctx.template.symbols, v)
                                .with_context(|| format!("Parsing as TDim: `{v}'"))?;
                            Ok(DimFact::from(dim))
                        }
                    }
                    _ => Ok(DimFact::default()),
                }
            })
            .collect::<TractResult<_>>()?;
        fact = fact.with_shape(ShapeFactoid::closed(shape));
    }
    Ok(fact)
}

fn get_external_resources(
    provider: &dyn ModelDataResolver,
    t: &TensorProto,
    path: &str,
) -> TractResult<Vec<u8>> {
    let mut tensor_data: Vec<u8> = Vec::new();
    trace!("number of external file needed for this tensor: {}", t.external_data.len());
    let location = t
        .external_data
        .iter()
        .find(|it| it.key == "location")
        .map(|it| it.value.as_str())
        .context("Could not find external data location")?;

    let offset: usize = t
        .external_data
        .iter()
        .find(|it| it.key == "offset")
        .map(|it| it.value.parse())
        .transpose()
        .context("Error while parsing offset value on external data description")?
        .unwrap_or(0);

    let length: Option<usize> = t
        .external_data
        .iter()
        .find(|it| it.key == "length")
        .map(|it| it.value.parse())
        .transpose()
        .context("Error while parsing length value on external data description")?;

    let p = PathBuf::from(path).join(location);

    trace!("external file detected: {p:?}, offset {offset:?}, length: {length:?}");
    provider.read_bytes_from_path(&mut tensor_data, &p, offset, length)?;
    trace!("external file loaded");
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

pub fn load_tensor(
    provider: &dyn ModelDataResolver,
    t: &TensorProto,
    path: Option<&str>,
) -> TractResult<Tensor> {
    let dt = DataType::from_i32(t.data_type).unwrap().try_into()?;
    let shape: Vec<usize> = t.dims.iter().map(|&i| i as usize).collect();
    // detect if the tensor is rather in an external file than inside the onnx file directly
    let is_external = t.data_location.is_some()
        && t.data_location == Some(tensor_proto::DataLocation::External as i32);
    if t.raw_data.len() > 0 {
        create_tensor(shape, dt, &t.raw_data)
    } else if is_external {
        if let Some(model_path) = path {
            // external files will be loaded and fed to the tensor if necessary
            let external_data = get_external_resources(provider, t, model_path)?;
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
            DatumType::F16 => Array::from_shape_vec(
                &*shape,
                t.int32_data.iter().map(|&x| f16::from_bits(x as u16)).collect(),
            )?
            .into(),
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
            _ => unimplemented!("FIXME, struct tensor loading: {:?}", dt),
        };
        Ok(it)
    }
}

pub fn proto_from_reader<R: ::std::io::Read>(mut r: R) -> TractResult<TensorProto> {
    let mut v = vec![];
    r.read_to_end(&mut v)?;
    let b = bytes::Bytes::from(v);
    TensorProto::decode(b).context("Can not parse protobuf input")
}
