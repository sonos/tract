use std::fs;
use std::io::Read;

use tract_core::ops::prelude::*;
use CliResult;

pub fn for_size(size: &str) -> CliResult<TensorFact> {
    let splits = size.split("x").collect::<Vec<_>>();

    if splits.len() < 1 {
        bail!("The <size> argument should be formatted as {size}x{...}x{type}.");
    }

    let (datum_type, shape) = splits.split_last().unwrap();

    let shape = shape
        .iter()
        .map(|s| match *s {
            "S" => Ok(TDim::stream()),         // Streaming dimension.
            _ => Ok(s.parse::<i32>()?.into()), // Regular dimension.
        }).collect::<TractResult<Vec<TDim>>>()?;

    if shape.iter().filter(|o| o.is_stream()).count() > 1 {
        bail!("The <size> argument doesn't support more than one streaming dimension.");
    }

    let datum_type = match datum_type.to_lowercase().as_str() {
        "f64" => DatumType::F64,
        "f32" => DatumType::F32,
        "i32" => DatumType::I32,
        "i8" => DatumType::I8,
        "u8" => DatumType::U8,
        _ => bail!("Type of the input should be f64, f32, i32, i8 or u8."),
    };

    Ok(TensorFact::dt_shape(datum_type, shape))
}

fn tensor_for_text_data(filename: &str) -> CliResult<DtArray> {
    let mut file = fs::File::open(filename)
        .map_err(|e| format!("Reading tensor from {}, {:?}", filename, e))?;
    let mut data = String::new();
    file.read_to_string(&mut data)?;

    let mut lines = data.lines();
    let proto = for_size(lines.next().ok_or("Empty data file")?)?;
    let shape = proto.shape.concretize().unwrap();

    let values = lines.flat_map(|l| l.split_whitespace()).collect::<Vec<_>>();

    // We know there is at most one streaming dimension, so we can deduce the
    // missing value with a simple division.
    let product: usize = shape
        .iter()
        .map(|o| o.to_integer().unwrap_or(1) as usize)
        .product();
    let missing = values.len() / product;

    macro_rules! for_type {
        ($t:ty) => {{
            let array =
                ::ndarray::Array::from_iter(values.iter().map(|v| v.parse::<$t>().unwrap()));

            array.into_shape(
                shape
                    .iter()
                    .map(|i| i.to_integer().unwrap_or(missing as i32) as usize)
                    .collect::<Vec<_>>(),
            )?
        }};
    }

    let tensor = match proto.datum_type.concretize().unwrap() {
        DatumType::F64 => for_type!(f64).into(),
        DatumType::F32 => for_type!(f32).into(),
        DatumType::I32 => for_type!(i32).into(),
        DatumType::I8 => for_type!(i8).into(),
        DatumType::U8 => for_type!(u8).into(),
        _ => unimplemented!(),
    };
    Ok(tensor)
}

/// Parses the `data` command-line argument.
fn for_data(filename: &str) -> CliResult<TensorFact> {
    let tensor = if filename.ends_with(".pb") {
        let mut file = fs::File::open(filename)?;
        ::tract_onnx::tensor::from_reader(file)?
    } else {
        tensor_for_text_data(filename)?
    };

    Ok(tensor.into())
}

pub fn for_string(value: &str) -> CliResult<TensorFact> {
    if value.starts_with("@") {
        for_data(&value[1..])
    } else {
        for_size(value)
    }
}

pub fn make_inputs(values: &[TensorFact]) -> CliResult<TVec<DtArray>> {
    values.iter().map(|v| tensor_for_fact(v, None)).collect()
}

pub fn tensor_for_fact(fact: &TensorFact, streaming_dim: Option<usize>) -> CliResult<DtArray> {
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
                .map(|d| {
                    d.to_integer()
                        .ok()
                        .map(|d| d as usize)
                        .or(streaming_dim)
                        .unwrap()
                }).collect(),
            fact.datum_type.concretize().unwrap(),
        ))
    }
}

/// Generates a random tensor of a given size and type.
pub fn random(sizes: Vec<usize>, datum_type: DatumType) -> DtArray {
    use rand;
    use std::iter::repeat_with;
    let len = sizes.iter().product();
    macro_rules! r {
        ($t:ty) => {
            repeat_with(|| rand::random::<$t>())
                .take(len)
                .collect::<Vec<_>>()
        };
    }

    match datum_type {
        DatumType::F64 => DtArray::f64s(&*sizes, &*r!(f64)),
        DatumType::F32 => DtArray::f32s(&*sizes, &*r!(f32)),
        DatumType::I32 => DtArray::i32s(&*sizes, &*r!(i32)),
        DatumType::I8 => DtArray::i8s(&*sizes, &*r!(i8)),
        DatumType::U8 => DtArray::u8s(&*sizes, &*r!(u8)),
        _ => unimplemented!("missing type"),
    }.unwrap()
}
