#[cfg(target_vendor = "apple")]
extern crate tract_metal;

extern crate tract_cuda;

use std::fmt::{Debug, Display};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use ndarray::{Data, Dimension, RawData};
use tract_extra::WithTractExtra;
use tract_libcli::annotations::Annotations;
use tract_libcli::profile::BenchLimits;
use tract_libcli::tensor::RunTensors;
use tract_nnef::internal::{Runtime as _, parse_tdim};
use tract_nnef::prelude::{
    Framework, IntoTValue, SymbolValues, TValue, TVec, Tensor, TractResult, TypedFact, TypedModel,
    TypedSimplePlan,
};
use tract_onnx::prelude::InferenceModelExt;
use tract_onnx_opl::WithOnnx;
use tract_pulse::WithPulse;
use tract_pulse::model::{PulsedModel, PulsedModelExt};
use tract_transformers::WithTractTransformers;

use tract_api::*;

pub mod prelude {
    pub use crate::{Model, Runnable, Runtime, Value, nnef, onnx};
    pub use ndarray;
    pub use tract_api::*;
}

/// Creates an instance of an NNEF framework and parser that can be used to load and dump NNEF models.
pub fn nnef() -> Result<Nnef> {
    Ok(Nnef(tract_nnef::nnef()))
}

pub fn onnx() -> Result<Onnx> {
    Ok(Onnx(tract_onnx::onnx()))
}

pub fn runtime_for_name(name: &str) -> Result<Runtime> {
    if let Some(rt) = tract_onnx::tract_core::runtime::runtime_for_name(name) {
        Ok(Runtime(rt))
    } else {
        anyhow::bail!("Runtime {name} not available")
    }
}

/// tract version tag
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub struct Nnef(tract_nnef::internal::Nnef);

impl Debug for Nnef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Nnef")
    }
}

impl NnefInterface for Nnef {
    type Model = Model;
    fn load(&self, path: impl AsRef<Path>) -> Result<Model> {
        let m = self.0.model_for_path(path)?.into_decluttered()?;
        Ok(Model(m))
    }

    fn transform_model(&self, model: &mut Self::Model, transform_spec: &str) -> Result<()> {
        if let Some(transform) = self.0.get_transform(transform_spec)? {
            transform.transform(&mut model.0)?;
            model.0.declutter()?;
        }
        Ok(())
    }

    fn enable_tract_core(&mut self) -> Result<()> {
        self.0.enable_tract_core();
        Ok(())
    }

    fn enable_tract_extra(&mut self) -> Result<()> {
        self.0.enable_tract_extra();
        Ok(())
    }

    fn enable_tract_transformers(&mut self) -> Result<()> {
        self.0.enable_tract_transformers();
        Ok(())
    }

    fn enable_onnx(&mut self) -> Result<()> {
        self.0.enable_onnx();
        Ok(())
    }

    fn enable_pulse(&mut self) -> Result<()> {
        self.0.enable_pulse();
        Ok(())
    }

    fn enable_extended_identifier_syntax(&mut self) -> Result<()> {
        self.0.allow_extended_identifier_syntax(true);
        Ok(())
    }

    fn write_model_to_dir(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        self.0.write_to_dir(&model.0, path)
    }

    fn write_model_to_tar(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let file = std::fs::File::create(path)?;
        self.0.write_to_tar(&model.0, file)?;
        Ok(())
    }

    fn write_model_to_tar_gz(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        self.0.write_to_tar(&model.0, gz)?;
        Ok(())
    }
}

pub struct Onnx(tract_onnx::Onnx);
impl OnnxInterface for Onnx {
    type InferenceModel = InferenceModel;
    fn load(&self, path: impl AsRef<Path>) -> Result<Self::InferenceModel> {
        Ok(InferenceModel(self.0.model_for_path(path)?))
    }
}

pub struct InferenceModel(tract_onnx::prelude::InferenceModel);
impl InferenceModelInterface for InferenceModel {
    type Model = Model;
    type InferenceFact = InferenceFact;

    fn input_count(&self) -> Result<usize> {
        Ok(self.0.inputs.len())
    }

    fn output_count(&self) -> Result<usize> {
        Ok(self.0.outputs.len())
    }

    fn input_name(&self, id: usize) -> Result<String> {
        let node = self.0.inputs[id].node;
        Ok(self.0.node(node).name.to_string())
    }

    fn output_name(&self, id: usize) -> Result<String> {
        let node = self.0.outputs[id].node;
        Ok(self.0.node(node).name.to_string())
    }

    fn set_output_names(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<()> {
        self.0.set_output_names(outputs)
    }

    fn input_fact(&self, id: usize) -> Result<InferenceFact> {
        Ok(InferenceFact(self.0.input_fact(id)?.clone()))
    }

    fn set_input_fact(
        &mut self,
        id: usize,
        fact: impl AsFact<Self, Self::InferenceFact>,
    ) -> Result<()> {
        let fact = fact.as_fact(self)?.0.clone();
        self.0.set_input_fact(id, fact)
    }

    fn output_fact(&self, id: usize) -> Result<InferenceFact> {
        Ok(InferenceFact(self.0.output_fact(id)?.clone()))
    }

    fn set_output_fact(
        &mut self,
        id: usize,
        fact: impl AsFact<Self, Self::InferenceFact>,
    ) -> Result<()> {
        let fact = fact.as_fact(self)?.0.clone();
        self.0.set_output_fact(id, fact)
    }

    fn analyse(&mut self) -> Result<()> {
        self.0.analyse(false)?;
        Ok(())
    }

    fn into_tract(self) -> Result<Self::Model> {
        let typed = self.0.into_typed()?.into_decluttered()?;
        Ok(Model(typed))
    }
}

// MODEL
pub struct Model(TypedModel);

impl ModelInterface for Model {
    type Fact = Fact;
    type Runnable = Runnable;
    type Value = Value;

    fn input_count(&self) -> Result<usize> {
        Ok(self.0.inputs.len())
    }

    fn output_count(&self) -> Result<usize> {
        Ok(self.0.outputs.len())
    }

    fn input_name(&self, id: usize) -> Result<String> {
        let node = self.0.inputs[id].node;
        Ok(self.0.node(node).name.to_string())
    }

    fn output_name(&self, id: usize) -> Result<String> {
        let node = self.0.outputs[id].node;
        Ok(self.0.node(node).name.to_string())
    }

    fn set_output_names(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<()> {
        self.0.set_output_names(outputs)
    }

    fn input_fact(&self, id: usize) -> Result<Fact> {
        Ok(Fact(self.0.input_fact(id)?.clone()))
    }

    fn output_fact(&self, id: usize) -> Result<Fact> {
        Ok(Fact(self.0.output_fact(id)?.clone()))
    }

    fn into_runnable(self) -> Result<Runnable> {
        let runnable = tract_nnef::internal::DefaultRuntime.prepare(self.0)?;
        Ok(Runnable(runnable.into()))
    }

    fn concretize_symbols(
        &mut self,
        values: impl IntoIterator<Item = (impl AsRef<str>, i64)>,
    ) -> Result<()> {
        let mut table = SymbolValues::default();
        for (k, v) in values {
            table = table.with(&self.0.symbols.sym(k.as_ref()), v);
        }
        self.0 = self.0.concretize_dims(&table)?;
        Ok(())
    }

    fn transform(&mut self, transform: &str) -> Result<()> {
        let transform = tract_onnx::tract_core::transform::get_transform(transform)
            .with_context(|| format!("transform `{transform}' could not be found"))?;
        transform.transform(&mut self.0)
    }

    fn pulse(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<()> {
        let stream_sym = self.0.symbols.sym(name.as_ref());
        let pulse_dim = parse_tdim(&self.0.symbols, value.as_ref())?;
        self.0 = PulsedModel::new(&self.0, stream_sym, &pulse_dim)?.into_typed()?;
        Ok(())
    }

    fn property_keys(&self) -> Result<Vec<String>> {
        Ok(self.0.properties.keys().cloned().collect())
    }

    fn property(&self, name: impl AsRef<str>) -> Result<Value> {
        let name = name.as_ref();
        self.0
            .properties
            .get(name)
            .with_context(|| format!("no property for name {name}"))
            .map(|t| Value(t.clone().into_tvalue()))
    }
}

// RUNTIME
pub struct Runtime(&'static dyn tract_nnef::internal::Runtime);

impl RuntimeInterface for Runtime {
    type Runnable = Runnable;

    type Model = Model;

    fn name(&self) -> Result<String> {
        Ok(self.0.name().into_owned())
    }

    fn prepare(&self, model: Self::Model) -> Result<Self::Runnable> {
        let runnable = self.0.prepare(model.0)?;
        Ok(Runnable(runnable.into()))
    }
}

// RUNNABLE
pub struct Runnable(Arc<dyn tract_nnef::internal::Runnable>);

impl RunnableInterface for Runnable {
    type Value = Value;
    type State = State;

    fn run<I, V, E>(&self, inputs: I) -> Result<Vec<Value>>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error>,
    {
        self.spawn_state()?.run(inputs)
    }

    fn spawn_state(&self) -> Result<State> {
        Ok(State(self.0.spawn()?))
    }

    fn input_count(&self) -> Result<usize> {
        Ok(self.0.input_count())
    }

    fn output_count(&self) -> Result<usize> {
        Ok(self.0.output_count())
    }

    fn cost_json(&self) -> Result<String> {
        let input: Option<Vec<Value>> = None;
        let states: Option<Vec<Value>> = None;
        self.profile_json(input, states)
    }

    fn profile_json<I, IV, IE, S, SV, SE>(
        &self,
        inputs: Option<I>,
        state_initializers: Option<S>,
    ) -> Result<String>
    where
        I: IntoIterator<Item = IV>,
        IV: TryInto<Self::Value, Error = IE>,
        IE: Into<anyhow::Error> + Debug,
        S: IntoIterator<Item = SV>,
        SV: TryInto<Self::Value, Error = SE>,
        SE: Into<anyhow::Error> + Debug,
    {
        let model = self
            .0
            .downcast_ref::<Arc<TypedSimplePlan>>()
            .context("Can only profile TypedModel-based runnables")?
            .model();
        let mut annotations = Annotations::from_model(model)?;
        tract_libcli::profile::extract_costs(&mut annotations, model, &SymbolValues::default())?;
        if let Some(inputs) = inputs {
            let inputs = inputs
                .into_iter()
                .map(|v| Ok(v.try_into().unwrap().0))
                .collect::<TractResult<TVec<_>>>()?;

            let mut state_inits: Vec<TValue> = vec![];

            if let Some(states) = state_initializers {
                states.into_iter().for_each(|s| state_inits.push(s.try_into().unwrap().0));
            }
            tract_libcli::profile::profile(
                &self.0,
                &BenchLimits::default(),
                &mut annotations,
                &RunTensors { sources: vec![inputs], state_initializers: state_inits },
                None,
                true,
            )?;
        };
        let export = tract_libcli::export::GraphPerfInfo::from(model, &annotations);
        Ok(serde_json::to_string(&export)?)
    }
}

// STATE
pub struct State(Box<dyn tract_nnef::internal::State>);

impl StateInterface for State {
    type Fact = Fact;
    type Value = Value;

    fn input_count(&self) -> Result<usize> {
        Ok(self.0.input_count())
    }

    fn output_count(&self) -> Result<usize> {
        Ok(self.0.output_count())
    }

    fn run<I, V, E>(&mut self, inputs: I) -> Result<Vec<Value>>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Value, Error = E>,
        E: Into<anyhow::Error>,
    {
        let inputs: TVec<TValue> = inputs
            .into_iter()
            .map(|i| i.try_into().map_err(|e| e.into()).map(|v| v.0))
            .collect::<Result<_>>()?;
        let outputs = self.0.run(inputs)?;
        Ok(outputs.into_iter().map(Value).collect())
    }

    fn initializable_states_count(&self) -> Result<usize> {
        Ok(self.0.initializable_states_count())
    }

    fn get_states_facts(&self) -> Result<Vec<Fact>> {
        Ok(self.0.get_states_facts().into_iter().map(Fact).collect())
    }

    fn set_states<I, V, E>(&mut self, state_initializers: I) -> Result<()>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error> + Debug,
    {
        let states: Vec<TValue> = state_initializers
            .into_iter()
            .map(|si| -> TractResult<TValue> {
                let v: Value =
                    si.try_into().map_err(|e| anyhow::anyhow!("Failed conversion:  {e:?}"))?;
                Ok(v.0)
            })
            .collect::<Result<Vec<TValue>>>()?;
        self.0.init_state(&states)?;
        Ok(())
    }

    fn get_states(&self) -> Result<Vec<Self::Value>> {
        Ok(self.0.get_states()?.into_iter().map(Value).collect())
    }
}

// VALUE
#[derive(Clone, Debug)]
pub struct Value(TValue);

impl ValueInterface for Value {
    fn from_bytes(dt: DatumType, shape: &[usize], data: &[u8]) -> Result<Self> {
        let dt = to_internal_dt(dt);
        let len = shape.iter().product::<usize>() * dt.size_of();
        anyhow::ensure!(len == data.len());
        let tensor = unsafe { Tensor::from_raw_dt(dt, shape, data)? };
        Ok(Value(tensor.into_tvalue()))
    }

    fn as_bytes(&self) -> Result<(DatumType, &[usize], &[u8])> {
        let dt = from_internal_dt(self.0.datum_type())?;
        Ok((dt, self.0.shape(), unsafe { self.0.as_slice_unchecked::<u8>() }))
    }
}

#[derive(Clone, Debug)]
pub struct Fact(TypedFact);

impl FactInterface for Fact {}

impl Fact {
    fn new(model: &mut Model, spec: impl ToString) -> Result<Fact> {
        let fact = tract_libcli::tensor::parse_spec(&model.0.symbols, &spec.to_string())?;
        let fact = tract_onnx::prelude::Fact::to_typed_fact(&fact)?.into_owned();
        Ok(Fact(fact))
    }

    fn dump(&self) -> Result<String> {
        Ok(format!("{:?}", self.0))
    }
}

impl Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dump().unwrap())
    }
}

#[derive(Default, Clone, Debug)]
pub struct InferenceFact(tract_onnx::prelude::InferenceFact);

impl InferenceFactInterface for InferenceFact {
    fn empty() -> Result<InferenceFact> {
        Ok(InferenceFact(Default::default()))
    }
}

impl InferenceFact {
    fn new(model: &mut InferenceModel, spec: impl ToString) -> Result<InferenceFact> {
        let fact = tract_libcli::tensor::parse_spec(&model.0.symbols, &spec.to_string())?;
        Ok(InferenceFact(fact))
    }

    fn dump(&self) -> Result<String> {
        Ok(format!("{:?}", self.0))
    }
}

impl Display for InferenceFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dump().unwrap())
    }
}

value_from_to_ndarray!();
as_inference_fact_impl!(InferenceModel, InferenceFact);
as_fact_impl!(Model, Fact);

/*
#[inline(always)]
fn to_datum_type<T: TractProxyDatumType>() -> Result<tract_nnef::prelude::DatumType> {
macro_rules! dt { ($($t:ty),*) => { $(if TypeId::of::<T>() == TypeId::of::<$t>() { return Ok(<$t>::datum_type()); })* }}
dt!(f32, f16, f64, i64, i32, i16, i8, bool, u64, u32, u16, u8);
anyhow::bail!("Unsupported type {}", std::any::type_name::<T>())
}
*/

fn to_internal_dt(it: DatumType) -> tract_nnef::prelude::DatumType {
    use DatumType::*;
    use tract_nnef::prelude::DatumType::*;
    match it {
        TRACT_DATUM_TYPE_BOOL => Bool,
        TRACT_DATUM_TYPE_U8 => U8,
        TRACT_DATUM_TYPE_U16 => U16,
        TRACT_DATUM_TYPE_U32 => U32,
        TRACT_DATUM_TYPE_U64 => U64,
        TRACT_DATUM_TYPE_I8 => I8,
        TRACT_DATUM_TYPE_I16 => I16,
        TRACT_DATUM_TYPE_I32 => I32,
        TRACT_DATUM_TYPE_I64 => I64,
        TRACT_DATUM_TYPE_F16 => F16,
        TRACT_DATUM_TYPE_F32 => F32,
        TRACT_DATUM_TYPE_F64 => F64,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_I16 => ComplexI16,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_I32 => ComplexI32,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_I64 => ComplexI64,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_F16 => ComplexF16,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_F32 => ComplexF32,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_F64 => ComplexF64,
    }
}

fn from_internal_dt(it: tract_nnef::prelude::DatumType) -> Result<DatumType> {
    use DatumType::*;
    use tract_nnef::prelude::DatumType::*;
    Ok(match it {
        Bool => TRACT_DATUM_TYPE_BOOL,
        U8 => TRACT_DATUM_TYPE_U8,
        U16 => TRACT_DATUM_TYPE_U16,
        U32 => TRACT_DATUM_TYPE_U32,
        U64 => TRACT_DATUM_TYPE_U64,
        I8 => TRACT_DATUM_TYPE_I8,
        I16 => TRACT_DATUM_TYPE_I16,
        I32 => TRACT_DATUM_TYPE_I32,
        I64 => TRACT_DATUM_TYPE_I64,
        F16 => TRACT_DATUM_TYPE_F16,
        F32 => TRACT_DATUM_TYPE_F32,
        F64 => TRACT_DATUM_TYPE_F64,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_I16 => ComplexI16,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_I32 => ComplexI32,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_I64 => ComplexI64,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_F16 => ComplexF16,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_F32 => ComplexF32,
        #[cfg(feature = "complex")]
        TRACT_DATUM_TYPE_COMPLEX_F64 => ComplexF64,
        _ => {
            anyhow::bail!("Unsupported DatumType in the public API {:?}", it)
        }
    })
}
