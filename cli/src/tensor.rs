use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::str::FromStr;
use std::sync::Mutex;

use crate::model::Model;
use crate::{CliResult, Parameters};
use tract_hir::internal::*;

fn parse_dt(dt: &str) -> CliResult<DatumType> {
    Ok(match dt.to_lowercase().as_ref() {
        "f16" => DatumType::F16,
        "f32" => DatumType::F32,
        "f64" => DatumType::F64,
        "i8" => DatumType::I8,
        "i16" => DatumType::I16,
        "i32" => DatumType::I32,
        "i64" => DatumType::I64,
        "u8" => DatumType::U8,
        "u16" => DatumType::U16,
        "u32" => DatumType::U32,
        "u64" => DatumType::U64,
        "tdim" => DatumType::TDim,
        _ => bail!(
            "Type of the input should be f16, f32, f64, i8, i16, i16, i32, u8, u16, u32, u64, TDim."
            ),
    })
}

pub fn parse_spec(size: &str) -> CliResult<InferenceFact> {
    if size.len() == 0 {
        return Ok(InferenceFact::default());
    }
    if size.contains("x") && !size.contains(",") {
        parse_x_spec(size)
    } else {
        parse_coma_spec(size)
    }
}

pub fn parse_coma_spec(size: &str) -> CliResult<InferenceFact> {
    let splits = size.split(",").collect::<Vec<_>>();

    if splits.len() < 1 {
        // Hide '{' in this error message from the formatting machinery in bail macro
        let msg = "The <size> argument should be formatted as {size},{...},{type}.";
        bail!(msg);
    }

    let last = splits.last().unwrap();
    let (datum_type, shape) = if let Ok(dt) = parse_dt(last) {
        (Some(dt), &splits[0..splits.len() - 1])
    } else {
        (None, &*splits)
    };

    let shape = ShapeFactoid::closed(
        shape
            .iter()
            .map(|&s| {
                Ok(if s == "_" { GenericFactoid::Any } else { GenericFactoid::Only(parse_dim(s)?) })
            })
            .collect::<CliResult<TVec<DimFact>>>()?,
    );

    if let Some(dt) = datum_type {
        Ok(InferenceFact::dt_shape(dt, shape))
    } else {
        Ok(InferenceFact::shape(shape))
    }
}

pub fn parse_dim(i: &str) -> CliResult<TDim> {
    // ensure the magic S is pre-registered
    #[cfg(feature = "pulse")]
    let _ = tract_pulse::internal::stream_symbol();

    if i.len() == 0 {
        bail!("Can not parse empty string as Dim")
    }
    let number_len = i.chars().take_while(|c| c.is_digit(10)).count();
    let symbol_len = i.len() - number_len;
    if symbol_len > 1 {
        bail!("Can not parse {} as Dim", i)
    }
    let number: i64 = if number_len > 0 { i[..number_len].parse()? } else { 1 };
    if symbol_len == 0 {
        return Ok(number.to_dim());
    }
    let symbol = i.chars().last().unwrap();
    let symbol = Symbol::from(symbol);
    Ok(symbol.to_dim() * number)
}

pub fn parse_x_spec(size: &str) -> CliResult<InferenceFact> {
    warn!(
        "Deprecated \"x\" syntax for shape : please use the comma as separator, x is now a symbol."
    );
    let splits = size.split("x").collect::<Vec<_>>();

    if splits.len() < 1 {
        // Hide '{' in this error message from the formatting machinery in bail macro
        let msg = "The <size> argument should be formatted as {size},{...},{type}.";
        bail!(msg);
    }

    let last = splits.last().unwrap();
    let (datum_type, shape) = if last.ends_with("S") || last.parse::<i32>().is_ok() {
        (None, &*splits)
    } else {
        let datum_type = parse_dt(splits.last().unwrap())?;
        (Some(datum_type), &splits[0..splits.len() - 1])
    };

    let shape = ShapeFactoid::closed(
        shape
            .iter()
            .map(|&s| {
                Ok(if s == "_" {
                    GenericFactoid::Any
                } else {
                    GenericFactoid::Only(parse_dim_stream(s)?)
                })
            })
            .collect::<CliResult<TVec<DimFact>>>()?,
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
        .map(|v| Ok(v.parse::<T>().map_err(|_| format_err!("Failed to parse {}", v))?))
        .collect::<CliResult<Vec<T>>>()?;
    Ok(tract_ndarray::Array::from_shape_vec(shape, values)?.into())
}

fn tensor_for_text_data(filename: &str) -> CliResult<Tensor> {
    let mut file = fs::File::open(filename)
        .map_err(|e| format_err!("Reading tensor from {}, {:?}", filename, e))?;
    let mut data = String::new();
    file.read_to_string(&mut data)?;

    let mut lines = data.lines();
    let proto = parse_spec(lines.next().context("Empty data file")?)?;
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
                fs::File::open(filename).with_context(|| format!("Can't open {:?}", filename))?;
            let proto = ::tract_onnx::tensor::proto_from_reader(file)?;
            Ok((
                Some(proto.name.to_string()).filter(|s| !s.is_empty()),
                Tensor::try_from(proto)?.into(),
            ))
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

pub fn for_npz(npz: &mut ndarray_npy::NpzReader<fs::File>, name: &str) -> CliResult<Tensor> {
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<f32>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<f64>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<i8>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<i16>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<i32>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<i64>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<u8>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<u16>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<u32>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<u64>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
    }
    if let Ok(t) = npz.by_name::<tract_ndarray::OwnedRepr<bool>, tract_ndarray::IxDyn>(name) {
        return Ok(t.into_tensor());
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
            let dt = spec
                .datum_type
                .concretize()
                .context("Must specify type when giving tensor value")?;
            let shape = spec
                .shape
                .as_concrete_finite()?
                .context("Must specify concrete shape when giving tensor value")?;
            let tensor = dispatch_datum!(parse_values(dt)(&*shape, value.collect()))?;
            Ok((name, tensor.into()))
        } else {
            Ok((name, parse_spec(value)?))
        }
    }
}

#[cfg(feature = "pulse")]
fn parse_dim_stream(s: &str) -> CliResult<TDim> {
    use tract_pulse::internal::stream_dim;
    if s == "S" {
        Ok(stream_dim())
    } else if s.ends_with("S") {
        let number: String = s.chars().take_while(|c| c.is_digit(10)).collect();
        let number: i64 = number.parse::<i64>().map(|i| i.into())?;
        Ok(stream_dim() * number)
    } else {
        Ok(s.parse::<i64>().map(|i| i.into())?)
    }
}

#[cfg(not(feature = "pulse"))]
fn parse_dim_stream(s: &str) -> CliResult<TDim> {
    Ok(s.parse::<i64>().map(|i| i.into())?)
}

lazy_static::lazy_static! {
    static ref WARNING_ONCE: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
}

fn warn_once(msg: String) {
    if WARNING_ONCE.lock().unwrap().insert(msg.clone()) {
        warn!("{}", msg);
    }
}

pub fn retrieve_or_make_inputs(
    tract: &dyn Model,
    params: &Parameters,
) -> CliResult<Vec<TVec<Tensor>>> {
    let mut tmp: TVec<Vec<Tensor>> = tvec![];
    for input in tract.input_outlets() {
        let name = tract.node_name(input.node);
        let fact = tract.outlet_typedfact(*input)?;
        if let Some(value) = params.input_values.get(name) {
            if TypedFact::from(value[0].clone()).compatible_with(&fact) {
                info!("Using fixed input for input called {} ({} turn(s))", name, value.len());
                tmp.push(value.iter().map(|t| t.clone().into_tensor()).collect())
            } else if fact.datum_type == f16::datum_type()
                && value[0].datum_type() == f32::datum_type()
                && params.allow_float_casts
            {
                tmp.push(value.iter().map(|t| t.cast_to::<f16>().unwrap().into_owned()).collect())
            } else if value.len() == 1
                && tract.properties().contains_key("pulse.delay")
                && tract.input_outlets().len() == 1
                && tract.output_outlets().len() == 1
            {
                let value = &value[0];
                let input_pulse_axis = tract
                    .properties()
                    .get("pulse.input_axes")
                    .context("Expect pulse.input_axes property")?
                    .cast_to::<i64>()?
                    .as_slice::<i64>()?[0] as usize;
                let input_pulse = fact.shape.get(input_pulse_axis).unwrap().to_usize().unwrap();
                let input_len = value.shape()[input_pulse_axis];

                let output_pulse_axis = tract
                    .properties()
                    .get("pulse.output_axes")
                    .context("Expect pulse.output_axes property")?
                    .cast_to::<i64>()?
                    .as_slice::<i64>()?[0] as usize;
                let output_fact = tract.outlet_typedfact(tract.output_outlets()[0])?;
                let output_pulse =
                    output_fact.shape.get(output_pulse_axis).unwrap().to_usize().unwrap();
                let output_len = input_len * output_pulse / input_pulse;
                let output_delay = tract.properties()["pulse.delay"].as_slice::<i64>()?[0] as usize;
                let last_frame = output_len + output_delay;
                let needed_pulses = last_frame.divceil(output_pulse);
                let mut values = vec![];
                for ix in 0..needed_pulses {
                    let mut t =
                        Tensor::zero_dt(fact.datum_type, fact.shape.as_concrete().unwrap())?;
                    let start = ix * input_pulse;
                    let end = (start + input_pulse).min(input_len);
                    if end > start {
                        t.assign_slice(0..end - start, value, start..end, input_pulse_axis)?;
                    }
                    values.push(t);
                }
                info!("Generated {} pulse of input", needed_pulses);
                tmp.push(values);
            } else {
                bail!("For input {}, can not reconcile model input fact {:?} with provided input {:?}", name, fact, value[0]);
            };
        } else if params.allow_random_input {
            let fact = tract.outlet_typedfact(*input)?;
            warn_once(format!("Using random input for input called {:?}: {:?}", name, fact));
            tmp.push(vec![crate::tensor::tensor_for_fact(&fact, None)?]);
        } else {
            bail!("Unmatched tensor {}. Fix the input or use \"--allow-random-input\" if this was intended", name);
        }
    }
    Ok((0..tmp[0].len()).map(|turn| tmp.iter().map(|t| t[turn].clone()).collect()).collect())
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

#[allow(unused_variables)]
pub fn tensor_for_fact(fact: &TypedFact, streaming_dim: Option<usize>) -> CliResult<Tensor> {
    if let Some(value) = &fact.konst {
        return Ok(value.clone().into_tensor());
    }
    #[cfg(pulse)]
    {
        if fact.shape.stream_info().is_some() {
            use tract_pulse::fact::StreamFact;
            use tract_pulse::internal::stream_symbol;
            let s = stream_symbol();
            if let Some(dim) = streaming_dim {
                let shape = fact
                    .shape
                    .iter()
                    .map(|d| {
                        d.eval(&SymbolValues::default().with(s, dim as i64)).to_usize().unwrap()
                    })
                    .collect::<TVec<_>>();
                return Ok(random(&shape, fact.datum_type));
            } else {
                bail!("random tensor requires a streaming dim")
            }
        }
    }
    Ok(random(
        &fact
            .shape
            .as_concrete()
            .with_context(|| format!("Expected concrete shape, found: {:?}", fact))?,
        fact.datum_type,
    ))
}

/// Generates a random tensor of a given size and type.
pub fn random(sizes: &[usize], datum_type: DatumType) -> Tensor {
    use std::iter::repeat_with;
    fn make<D>(shape: &[usize]) -> Tensor
    where
        D: Datum,
        rand::distributions::Standard: rand::distributions::Distribution<D>,
    {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(21242);
        let len = shape.iter().product();
        tract_core::ndarray::ArrayD::from_shape_vec(
            shape,
            repeat_with(|| rng.gen::<D>()).take(len).collect(),
        )
        .unwrap()
        .into()
    }
    use DatumType::*;
    let mut t = match datum_type {
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
        QU8(_) => make::<u8>(sizes),
        QI8(_) => make::<i8>(sizes),
        QI32(_) => make::<i32>(sizes),
        _ => panic!("Can generate random tensor for {:?}", datum_type),
    };
    unsafe {
        t.set_datum_type(datum_type);
    }
    t
}
