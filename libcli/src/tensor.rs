use std::collections::HashSet;
use std::io::{Read, Seek};
use std::ops::Range;
use std::str::FromStr;
use std::sync::Mutex;

use crate::model::Model;
use tract_hir::internal::*;
use tract_num_traits::Zero;

#[derive(Debug, Default, Clone)]
pub struct TensorsValues(pub Vec<TensorValues>);

impl TensorsValues {
    pub fn by_name(&self, name: &str) -> Option<&TensorValues> {
        self.0.iter().find(|t| t.name.as_deref() == Some(name))
    }
    pub fn by_name_mut(&mut self, name: &str) -> Option<&mut TensorValues> {
        self.0.iter_mut().find(|t| t.name.as_deref() == Some(name))
    }
    pub fn by_name_mut_with_default(&mut self, name: &str) -> &mut TensorValues {
        if self.by_name_mut(name).is_none() {
            self.add(TensorValues { name: Some(name.to_string()), ..TensorValues::default() });
        }
        self.by_name_mut(name).unwrap()
    }

    pub fn by_input_ix(&self, ix: usize) -> Option<&TensorValues> {
        self.0.iter().find(|t| t.input_index == Some(ix))
    }
    pub fn by_input_ix_mut(&mut self, ix: usize) -> Option<&mut TensorValues> {
        self.0.iter_mut().find(|t| t.input_index == Some(ix))
    }
    pub fn by_input_ix_mut_with_default(&mut self, ix: usize) -> &mut TensorValues {
        if self.by_input_ix_mut(ix).is_none() {
            self.add(TensorValues { input_index: Some(ix), ..TensorValues::default() });
        }
        self.by_input_ix_mut(ix).unwrap()
    }

    pub fn add(&mut self, other: TensorValues) {
        let mut tensor = other.input_index.and_then(|ix| self.by_input_ix_mut(ix));

        if tensor.is_none() {
            tensor = other.name.as_deref().and_then(|ix| self.by_name_mut(ix))
        }

        if let Some(tensor) = tensor {
            if tensor.fact.is_none() {
                tensor.fact = other.fact;
            }
            if tensor.values.is_none() {
                tensor.values = other.values;
            }
        } else {
            self.0.push(other.clone());
        };
    }

    pub fn input_by_name(&self, name: &str) -> Option<&TensorValues> {
        self.0
            .iter()
            .filter(|tv| tv.output_index.is_none() && !tv.only_output)
            .find(|t| t.name.as_deref() == Some(name))
    }
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct TensorValues {
    pub input_index: Option<usize>,
    pub output_index: Option<usize>,
    pub name: Option<String>,
    pub fact: Option<InferenceFact>,
    pub values: Option<Vec<TValue>>,
    pub random_range: Option<Range<f32>>,
    pub only_input: bool,
    pub only_output: bool,
}

fn parse_dt(dt: &str) -> TractResult<DatumType> {
    Ok(match dt.to_lowercase().as_ref() {
        "bool" => DatumType::Bool,
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

pub fn parse_spec(symbol_table: &SymbolScope, size: &str) -> TractResult<InferenceFact> {
    if size.is_empty() {
        return Ok(InferenceFact::default());
    }
    parse_coma_spec(symbol_table, size)
}

pub fn parse_coma_spec(symbol_table: &SymbolScope, size: &str) -> TractResult<InferenceFact> {
    let splits = size.split(',').collect::<Vec<_>>();

    #[allow(clippy::literal_string_with_formatting_args)]
    if splits.is_empty() {
        bail!("The <size> argument should be formatted as {{size}},{{...}},{{type}}.");
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
                Ok(if s == "_" {
                    GenericFactoid::Any
                } else {
                    GenericFactoid::Only(parse_tdim(symbol_table, s)?)
                })
            })
            .collect::<TractResult<TVec<DimFact>>>()?,
    );

    if let Some(dt) = datum_type {
        Ok(InferenceFact::dt_shape(dt, shape))
    } else {
        Ok(InferenceFact::shape(shape))
    }
}

fn parse_values<T: Datum + FromStr>(shape: &[usize], it: Vec<&str>) -> TractResult<Tensor> {
    let values = it
        .into_iter()
        .map(|v| v.parse::<T>().map_err(|_| format_err!("Failed to parse {}", v)))
        .collect::<TractResult<Vec<T>>>()?;
    Ok(tract_ndarray::Array::from_shape_vec(shape, values)?.into())
}

fn tensor_for_text_data(
    symbol_table: &SymbolScope,
    _filename: &str,
    mut reader: impl Read,
) -> TractResult<Tensor> {
    let mut data = String::new();
    reader.read_to_string(&mut data)?;

    let mut lines = data.lines();
    let proto = parse_spec(symbol_table, lines.next().context("Empty data file")?)?;
    let shape = proto.shape.concretize().unwrap();

    let values = lines.flat_map(|l| l.split_whitespace()).collect::<Vec<&str>>();

    // We know there is at most one streaming dimension, so we can deduce the
    // missing value with a simple division.
    let product: usize = shape.iter().map(|o| o.to_usize().unwrap_or(1)).product();
    let missing = values.len() / product;

    let shape: Vec<_> = shape.iter().map(|d| d.to_usize().unwrap_or(missing)).collect();
    dispatch_numbers!(parse_values(proto.datum_type.concretize().unwrap())(&*shape, values))
}

/// Parses the `data` command-line argument.
pub fn for_data(
    symbol_table: &SymbolScope,
    filename: &str,
    reader: impl Read + std::io::Seek,
) -> TractResult<(Option<String>, InferenceFact)> {
    #[allow(unused_imports)]
    use std::convert::TryFrom;
    if filename.ends_with(".pb") {
        #[cfg(feature = "onnx")]
        {
            use tract_onnx::data_resolver::FopenDataResolver;
            use tract_onnx::tensor::load_tensor;
            let proto = ::tract_onnx::tensor::proto_from_reader(reader)?;
            let tensor = load_tensor(&FopenDataResolver, &proto, None)?;
            Ok((Some(proto.name.to_string()).filter(|s| !s.is_empty()), tensor.into()))
        }
        #[cfg(not(feature = "onnx"))]
        {
            panic!("Loading tensor from protobuf requires onnx features");
        }
    } else if filename.contains(".npz:") {
        let mut tokens = filename.split(':');
        let (_filename, inner) = (tokens.next().unwrap(), tokens.next().unwrap());
        let mut npz = ndarray_npy::NpzReader::new(reader)?;
        Ok((None, for_npz(&mut npz, inner)?.into()))
    } else {
        Ok((None, tensor_for_text_data(symbol_table, filename, reader)?.into()))
    }
}

pub fn for_npz(
    npz: &mut ndarray_npy::NpzReader<impl Read + Seek>,
    name: &str,
) -> TractResult<Tensor> {
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

pub fn for_string(
    symbol_table: &SymbolScope,
    value: &str,
) -> TractResult<(Option<String>, InferenceFact)> {
    let (name, value) = if value.contains(':') {
        let mut splits = value.split(':');
        (Some(splits.next().unwrap().to_string()), splits.next().unwrap())
    } else {
        (None, value)
    };
    if value.contains('=') {
        let mut split = value.split('=');
        let spec = parse_spec(symbol_table, split.next().unwrap())?;
        let value = split.next().unwrap().split(',');
        let dt =
            spec.datum_type.concretize().context("Must specify type when giving tensor value")?;
        let shape = spec
            .shape
            .as_concrete_finite()?
            .context("Must specify concrete shape when giving tensor value")?;
        let tensor = if dt == TDim::datum_type() {
            let mut tensor = Tensor::zero::<TDim>(&shape)?;
            let values =
                value.map(|v| parse_tdim(symbol_table, v)).collect::<TractResult<Vec<_>>>()?;
            tensor.as_slice_mut::<TDim>()?.iter_mut().zip(values).for_each(|(t, v)| *t = v);
            tensor
        } else {
            dispatch_numbers!(parse_values(dt)(&*shape, value.collect()))?
        };
        Ok((name, tensor.into()))
    } else {
        Ok((name, parse_spec(symbol_table, value)?))
    }
}

lazy_static::lazy_static! {
    static ref MESSAGE_ONCE: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
}

fn info_once(msg: String) {
    if MESSAGE_ONCE.lock().unwrap().insert(msg.clone()) {
        info!("{msg}");
    }
}

pub struct RunParams {
    pub tensors_values: TensorsValues,
    pub allow_random_input: bool,
    pub allow_float_casts: bool,
    pub symbols: SymbolValues,
}

pub struct RunTensors {
    pub sources: Vec<TVec<TValue>>,
    pub state_initializers: Vec<TValue>,
}

fn get_or_make_tensors(
    model: &dyn Model,
    params: &RunParams,
    fact: TypedFact,
    name: &str,
    input_idx: usize,
    target: &mut TVec<Vec<TValue>>,
) -> TractResult<()> {
    if let Some(mut value) = params
        .tensors_values
        .by_name(name)
        .or_else(|| params.tensors_values.by_input_ix(input_idx))
        .and_then(|t| t.values.clone())
    {
        if !value[0].datum_type().is_quantized()
            && fact.datum_type.is_quantized()
            && value[0].datum_type() == fact.datum_type.unquantized()
        {
            value = value
                .iter()
                .map(|v| {
                    let mut v = v.clone().into_tensor();
                    unsafe { v.set_datum_type(fact.datum_type) };
                    v.into()
                })
                .collect();
        }
        if TypedFact::shape_and_dt_of(&value[0]).compatible_with(&fact) {
            info!("Using fixed input for input called {} ({} turn(s))", name, value.len());
            target.push(value.iter().map(|t| t.clone().into_tensor().into()).collect());
        } else if fact.datum_type == f16::datum_type()
            && value[0].datum_type() == f32::datum_type()
            && params.allow_float_casts
        {
            info!("Casting input to F16 for input called {} ({} turn(s))", name, value.len());
            target.push(
                value.iter().map(|t| t.cast_to::<f16>().unwrap().into_owned().into()).collect(),
            );
        } else if value.len() == 1 && model.properties().contains_key("pulse.delay") {
            let value = &value[0];
            let input_pulse_axis = model
                .properties()
                .get("pulse.input_axes")
                .context("Expect pulse.input_axes property")?
                .cast_to::<i64>()?
                .as_slice::<i64>()?[input_idx] as usize;
            let input_pulse = fact.shape.get(input_pulse_axis).unwrap().to_usize().unwrap();
            let input_len = value.shape()[input_pulse_axis];

            // how many pulses do we need to push full result out ?
            // guess by looking at len and delay of the first output
            let output_pulse_axis = model
                .properties()
                .get("pulse.output_axes")
                .context("Expect pulse.output_axes property")?
                .cast_to::<i64>()?
                .as_slice::<i64>()?[0] as usize;
            let output_fact = model.outlet_typedfact(model.output_outlets()[0])?;
            let output_pulse =
                output_fact.shape.get(output_pulse_axis).unwrap().to_usize().unwrap();
            let output_len = input_len * output_pulse / input_pulse;
            let output_delay = model.properties()["pulse.delay"].as_slice::<i64>()?[0] as usize;
            let last_frame = output_len + output_delay;
            let needed_pulses = last_frame.divceil(output_pulse);
            let mut values = vec![];
            for ix in 0..needed_pulses {
                let mut t = Tensor::zero_dt(fact.datum_type, fact.shape.as_concrete().unwrap())?;
                let start = ix * input_pulse;
                let end = (start + input_pulse).min(input_len);
                if end > start {
                    t.assign_slice(0..end - start, value, start..end, input_pulse_axis)?;
                }
                values.push(t.into());
            }
            info!(
                "Generated {} pulses of shape {:?} for input {}.",
                needed_pulses, fact.shape, input_idx
            );
            target.push(values);
        } else {
            bail!(
                "For input {}, can not reconcile model input fact {:?} with provided input {:?}",
                name,
                fact,
                value[0]
            );
        };
    } else if fact.shape.is_concrete() && fact.shape.volume() == TDim::zero() {
        let shape = fact.shape.as_concrete().unwrap();
        let tensor = Tensor::zero_dt(fact.datum_type, shape)?;
        target.push(vec![tensor.into()]);
    } else if params.allow_random_input {
        info_once(format!("Using random input for input called {name:?}: {fact:?}"));
        let tv = params
            .tensors_values
            .by_name(name)
            .or_else(|| params.tensors_values.by_input_ix(input_idx));
        let mut fact = fact.clone();
        fact.shape = fact.shape.iter().map(|dim| dim.eval(&params.symbols)).collect();
        target.push(vec![tensor_for_fact(&fact, None, tv)?.into()]);
    } else {
        bail!(
            "Unmatched tensor {}. Fix the input or use \"--allow-random-input\" if this was intended",
            name
        );
    }
    Ok(())
}

pub fn get_or_make_inputs(tract: &dyn Model, params: &RunParams) -> TractResult<RunTensors> {
    // Resolve source inputs
    let mut tmp_inputs = tvec![];
    for (ix, input) in tract.input_outlets().iter().enumerate() {
        let fact = tract.outlet_typedfact(*input)?;
        let name = tract.node_name(input.node);
        get_or_make_tensors(tract, params, fact, name, ix, &mut tmp_inputs)?;
    }

    let n_turns = tmp_inputs.first().map_or(0, |t| t.len());
    let sources = (0..n_turns)
        .map(|i| tmp_inputs.iter().map(|t| t[i].clone()).collect::<TVec<_>>())
        .collect::<Vec<_>>();

    // Resolve state initializers (KV Cache, etc.)
    let mut dummy_session_state = SessionState::default();
    let state_initializers = (0..tract.nodes_len())
        .filter_map(|id| {
            tract
                .node_op(id)
                .state(&mut dummy_session_state, id)
                .ok()
                .flatten()
                .and_then(|state| state.init_tensor_fact())
                .map(|(name, fact)| {
                    let mut tmp = tvec![];
                    get_or_make_tensors(tract, params, fact, &name, usize::MAX, &mut tmp)?;
                    Ok(tmp.remove(0).remove(0))
                })
        })
        .collect::<TractResult<Vec<_>>>()?;

    Ok(RunTensors { sources, state_initializers })
}

fn make_inputs(values: &[impl std::borrow::Borrow<TypedFact>]) -> TractResult<TVec<TValue>> {
    values.iter().map(|v| tensor_for_fact(v.borrow(), None, None).map(|t| t.into())).collect()
}

pub fn make_inputs_for_model(model: &dyn Model) -> TractResult<TVec<TValue>> {
    make_inputs(
        &model
            .input_outlets()
            .iter()
            .map(|&t| model.outlet_typedfact(t))
            .collect::<TractResult<Vec<TypedFact>>>()?,
    )
}

#[allow(unused_variables)]
pub fn tensor_for_fact(
    fact: &TypedFact,
    streaming_dim: Option<usize>,
    tv: Option<&TensorValues>,
) -> TractResult<Tensor> {
    if let Some(value) = &fact.konst {
        return Ok(value.clone().into_tensor());
    }
    Ok(random(
        fact.shape
            .as_concrete()
            .with_context(|| format!("Expected concrete shape, found: {fact:?}"))?,
        fact.datum_type,
        tv,
    ))
}

/// Generates a random tensor of a given size and type.
pub fn random(sizes: &[usize], datum_type: DatumType, tv: Option<&TensorValues>) -> Tensor {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(21242);
    let mut tensor = Tensor::zero::<f32>(sizes).unwrap();
    let slice = tensor.as_slice_mut::<f32>().unwrap();
    if let Some(range) = tv.and_then(|tv| tv.random_range.as_ref()) {
        slice.iter_mut().for_each(|x| *x = rng.gen_range(range.clone()))
    } else {
        slice.iter_mut().for_each(|x| *x = rng.r#gen())
    };
    tensor.cast_to_dt(datum_type).unwrap().into_owned()
}
