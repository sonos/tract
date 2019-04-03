use std::fs;
use std::io::Read;
use std::str::FromStr;

use crate::CliResult;
use tract_core::ops::prelude::*;

pub fn parse_spec(size: &str) -> CliResult<TensorFact> {
    let splits = size.split("x").collect::<Vec<_>>();

    if splits.len() < 1 {
        bail!("The <size> argument should be formatted as {size}x{...}x{type}.");
    }

    let (datum_type, shape) = if splits.last().unwrap().parse::<i32>().is_ok() {
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

    let shape = shape.iter().map(|s| Ok(s.parse()?)).collect::<TractResult<Vec<TDim>>>()?;

    if shape.iter().filter(|o| o.is_stream()).count() > 1 {
        bail!("The <size> argument doesn't support more than one streaming dimension.");
    }

    if let Some(dt) = datum_type {
        Ok(TensorFact::dt_shape(dt, shape))
    } else {
        Ok(TensorFact::shape(shape))
    }
}

fn parse_values<'a, T: Datum + FromStr>(shape: &[usize], it: Vec<&'a str>) -> CliResult<Tensor> {
    let values = it
        .into_iter()
        .map(|v| Ok(v.parse::<T>().map_err(|_| format!("Failed to parse {}", v))?))
        .collect::<CliResult<Vec<T>>>()?;
    Ok(::ndarray::Array::from_shape_vec(shape, values)?.into())
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
    let product: usize = shape.iter().map(|o| o.to_integer().unwrap_or(1) as usize).product();
    let missing = values.len() / product;

    let shape: Vec<_> =
        shape.iter().map(|d| d.to_integer().map(|i| i as usize).unwrap_or(missing)).collect();
    dispatch_copy!(parse_values(proto.datum_type.concretize().unwrap())(&*shape, values))
}

/// Parses the `data` command-line argument.
fn for_data(filename: &str) -> CliResult<TensorFact> {
    let tensor = if filename.ends_with(".pb") {
        #[cfg(feature = "onnx")]
        {
            let file = fs::File::open(filename)?;
            ::tract_onnx::tensor::from_reader(file)?
        }
        #[cfg(not(feature = "onnx"))]
        {
            panic!("Loading tensor from protobuf requires onnx features");
        }
    } else {
        tensor_for_text_data(filename)?
    };

    Ok(tensor.into())
}

pub fn for_string(value: &str) -> CliResult<TensorFact> {
    if value.starts_with("@") {
        for_data(&value[1..])
    } else {
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
            let tensor = dispatch_copy!(parse_values(dt)(&*shape, value.collect()))?;
            Ok(tensor.into())
        } else {
            parse_spec(value)
        }
    }
}

pub fn make_inputs(values: &[TensorFact]) -> CliResult<TVec<Tensor>> {
    values.iter().map(|v| tensor_for_fact(v, None)).collect()
}

pub fn tensor_for_fact(fact: &TensorFact, streaming_dim: Option<usize>) -> CliResult<Tensor> {
    if let Some(value) = fact.concretize() {
        Ok(value.as_tensor().to_owned())
    } else {
        if fact.stream_info()?.is_some() && streaming_dim.is_none() {
            Err("random tensor requires a streaming dim")?
        }
        Ok(random(
            fact.shape
                .concretize()
                .unwrap()
                .iter()
                .map(|d| d.to_integer().ok().map(|d| d as usize).or(streaming_dim).unwrap())
                .collect(),
            fact.datum_type.concretize().unwrap(),
        ))
    }
}

/// Generates a random tensor of a given size and type.
pub fn random(sizes: Vec<usize>, datum_type: DatumType) -> Tensor {
    use rand;
    use std::iter::repeat_with;
    fn make<D>(shape: Vec<usize>) -> Tensor
    where
        D: Datum,
        rand::distributions::Standard: rand::distributions::Distribution<D>,
    {
        let len = shape.iter().product();
        ndarray::ArrayD::from_shape_vec(
            shape,
            repeat_with(|| rand::random::<D>()).take(len).collect(),
        )
        .unwrap()
        .into()
    }

    match datum_type {
        DatumType::F64 => make::<f64>(sizes),
        DatumType::F32 => make::<f32>(sizes),
        DatumType::I32 => make::<i32>(sizes),
        DatumType::I8 => make::<i8>(sizes),
        DatumType::U8 => make::<u8>(sizes),
        _ => unimplemented!("missing type"),
    }
}
