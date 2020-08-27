use std::fs;
use std::io::Read;
use std::str::FromStr;

use crate::model::Model;
use crate::CliResult;
use tract_hir::internal::*;

pub fn parse_spec(size: &str) -> CliResult<InferenceFact> {
    if size.len() == 0 {
        return Ok(InferenceFact::default());
    }
    let splits = size.split("x").collect::<Vec<_>>();

    if splits.len() < 1 {
        bail!("The <size> argument should be formatted as {size}x{...}x{type}.");
    }

    let last = splits.last().unwrap();
    let (datum_type, shape) = if last.ends_with("S") || last.parse::<i32>().is_ok() {
        (None, &*splits)
    } else {
        let datum_type = match splits.last().unwrap().to_lowercase().as_str() {
            "f64" => DatumType::F64,
            "f32" => DatumType::F32,
            "i32" => DatumType::I32,
            "i8" => DatumType::I8,
            "u8" => DatumType::U8,
            _ => bail!("Type of the input should be f64, f32, i32, i8 or u8."),
        };
        (Some(datum_type), &splits[0..splits.len() - 1])
    };

    let shape = ShapeFactoid::closed(
        shape
            .iter()
            .map(|&s| {
                Ok(if s == "_" { GenericFactoid::Any } else { GenericFactoid::Only(s.parse()?) })
            })
            .collect::<TractResult<TVec<DimFact>>>()?,
    );

    if let Some(dt) = datum_type {
        Ok(InferenceFact::dt_shape(dt, shape))
    } else {
        Ok(InferenceFact::shape(shape))
    }
}

fn parse_values<'a, T: Datum + FromStr>(shape: &[usize], it: Vec<&'a str>) -> CliResult<Tensor> {
    let values = it
        .into_iter()
        .map(|v| Ok(v.parse::<T>().map_err(|_| format!("Failed to parse {}", v))?))
        .collect::<CliResult<Vec<T>>>()?;
    Ok(tract_ndarray::Array::from_shape_vec(shape, values)?.into())
}

fn tensor_for_text_data(filename: &str) -> CliResult<Tensor> {
    let mut file = fs::File::open(filename)
        .map_err(|e| format!("Reading tensor from {}, {:?}", filename, e))?;
    let mut data = String::new();
    file.read_to_string(&mut data)?;

    let mut lines = data.lines();
    let proto = parse_spec(lines.next().ok_or("Empty data file")?)?;
    let shape = proto.shape.concretize().unwrap();

    let values = lines.flat_map(|l| l.split_whitespace()).collect::<Vec<&str>>();

    // We know there is at most one streaming dimension, so we can deduce the
    // missing value with a simple division.
    let product: usize = shape.iter().map(|o| o.to_usize().unwrap_or(1)).product();
    let missing = values.len() / product;

    let shape: Vec<_> = shape.iter().map(|d| d.to_usize().unwrap_or(missing)).collect();
    dispatch_datum!(parse_values(proto.datum_type.concretize().unwrap())(&*shape, values))
}

/// Parses the `data` command-line argument.
pub fn for_data(filename: &str) -> CliResult<(Option<String>, InferenceFact)> {
    #[allow(unused_imports)]
    use std::convert::TryFrom;
    if filename.ends_with(".pb") {
        #[cfg(feature = "onnx")]
        {
            let file =
                fs::File::open(filename).chain_err(|| format!("Can't open {:?}", filename))?;
            let proto = ::tract_onnx::tensor::proto_from_reader(file)?;
            Ok((Some(proto.name.to_string()), Tensor::try_from(proto)?.into()))
        }
        #[cfg(not(feature = "onnx"))]
        {
            panic!("Loading tensor from protobuf requires onnx features");
        }
    } else if filename.contains(".npz:") {
        let mut tokens = filename.split(":");
        let (filename, inner) = (tokens.next().unwrap(), tokens.next().unwrap());
        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(filename)?)?;
        Ok((None, for_npz(&mut npz, inner)?.into()))
    } else {
        Ok((None, tensor_for_text_data(filename)?.into()))
    }
}

pub fn for_npz(npz: &mut ndarray_npy::NpzReader<fs::File>, name: &str) -> TractResult<Tensor> {
    fn rewrap<T: Datum>(array: tract_ndarray::ArrayD<T>) -> Tensor {
        let shape = array.shape().to_vec();
        unsafe {
            let vec = array.into_raw_vec();
            tract_core::ndarray::ArrayD::from_shape_vec_unchecked(shape, vec).into_tensor()
        }
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<f32>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<f64>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<i8>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<i16>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<i32>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<i64>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<u8>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<u16>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<u32>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<u64>, tract_ndarray::IxDyn>(name) {
        return Ok(rewrap(t));
    }
    bail!("Can not extract tensor from {}", name);
}

pub fn for_string(value: &str) -> CliResult<(Option<String>, InferenceFact)> {
    if value.starts_with("@") {
        for_data(&value[1..])
    } else {
        let (name, value) = if value.contains(":") {
            let mut splits = value.split(":");
            (Some(splits.next().unwrap().to_string()), splits.next().unwrap())
        } else {
            (None, value)
        };
        if value.contains("=") {
            let mut split = value.split("=");
            let spec = parse_spec(split.next().unwrap())?;
            let value = split.next().unwrap().split(",");
            let dt =
                spec.datum_type.concretize().ok_or("Must specify type when giving tensor value")?;
            let shape = spec
                .shape
                .as_concrete_finite()?
                .ok_or("Must specify concrete shape when giving tensor value")?;
            let tensor = dispatch_datum!(parse_values(dt)(&*shape, value.collect()))?;
            Ok((name, tensor.into()))
        } else {
            Ok((name, parse_spec(value)?))
        }
    }
}

pub fn make_inputs(values: &[impl std::borrow::Borrow<TypedFact>]) -> CliResult<TVec<Tensor>> {
    values.iter().map(|v| tensor_for_fact(v.borrow(), None)).collect()
}

pub fn make_inputs_for_model(model: &dyn Model) -> CliResult<TVec<Tensor>> {
    Ok(make_inputs(
        &*model
            .input_outlets()
            .iter()
            .map(|&t| model.outlet_typedfact(t))
            .collect::<TractResult<Vec<TypedFact>>>()?,
    )?)
}

pub fn tensor_for_fact(fact: &TypedFact, streaming_dim: Option<usize>) -> CliResult<Tensor> {
    use tract_core::pulse::{stream_symbol, StreamFact};
    let s = stream_symbol();
    if let Some(value) = &fact.konst {
        Ok(value.clone().into_tensor())
    } else if fact.shape.stream_info().is_some() {
        if let Some(dim) = streaming_dim {
            let shape = fact
                .shape
                .iter()
                .map(|d| d.eval(&hashmap!(s => dim as i64)).to_usize().unwrap())
                .collect::<TVec<_>>();
            Ok(random(&shape, fact.datum_type))
        } else {
            bail!("random tensor requires a streaming dim")
        }
    } else {
        Ok(random(&fact.shape.as_finite().unwrap(), fact.datum_type))
    }
}

/// Generates a random tensor of a given size and type.
pub fn random(sizes: &[usize], datum_type: DatumType) -> Tensor {
    use std::iter::repeat_with;
    fn make<D>(shape: &[usize]) -> Tensor
    where
        D: Datum,
        rand::distributions::Standard: rand::distributions::Distribution<D>,
    {
        let len = shape.iter().product();
        tract_core::ndarray::ArrayD::from_shape_vec(
            shape,
            repeat_with(|| rand::random::<D>()).take(len).collect(),
        )
        .unwrap()
        .into()
    }
    use DatumType::*;
    match datum_type {
        Bool => make::<bool>(sizes),
        I8 => make::<i8>(sizes),
        I16 => make::<i16>(sizes),
        I32 => make::<i32>(sizes),
        I64 => make::<i64>(sizes),
        U8 => make::<u8>(sizes),
        U16 => make::<u16>(sizes),
        F16 => make::<f32>(sizes).cast_to::<f16>().unwrap().into_owned(),
        F32 => make::<f32>(sizes),
        F64 => make::<f64>(sizes),
        _ => panic!("Can generate random tensor for {:?}", datum_type),
    }
}
