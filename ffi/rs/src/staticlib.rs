use std::fmt::{Debug, Display};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use ndarray::{Data, Dimension, RawData};
use tract_libcli::annotations::Annotations;
use tract_libcli::profile::BenchLimits;
use tract_nnef::internal::parse_tdim;
use tract_nnef::prelude::translator::Translate;
use tract_nnef::prelude::{
    tensor1, Datum, Framework, IntoTValue, SymbolValues, TValue, TVec, TractResult, TypedFact,
    TypedModel, TypedRunnableModel, TypedSimplePlan, TypedSimpleState,
};
use tract_onnx::prelude::InferenceModelExt;
use tract_onnx_opl::WithOnnx;
use tract_pulse::model::{PulsedModel, PulsedModelExt};

use crate::*;

pub struct Tract;
impl TractInterface for Tract {
    type Nnef = Nnef;
    type Onnx = Onnx;
    fn nnef() -> Result<Self::Nnef> {
        Ok(Nnef(tract_nnef::nnef()))
    }
    fn onnx() -> Result<Self::Onnx> {
        Ok(Onnx(tract_onnx::onnx()))
    }
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

// NNEF
pub struct Nnef(tract_nnef::internal::Nnef);

impl NnefInterface for Nnef {
    type Model = Model;
    fn model_for_path(&self, path: impl AsRef<Path>) -> Result<Model> {
        self.0.model_for_path(path).map(Model)
    }

    fn with_tract_core(self) -> Result<Nnef> {
        Ok(Nnef(self.0.with_tract_core()))
    }

    fn with_onnx(self) -> Result<Nnef> {
        Ok(Nnef(self.0.with_onnx()))
    }

    fn with_extended_identifier_syntax(mut self) -> Result<Nnef> {
        self.0.allow_extended_identifier_syntax(true);
        Ok(self)
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
    fn model_for_path(&self, path: impl AsRef<Path>) -> Result<Self::InferenceModel> {
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

    fn into_typed(self) -> Result<Self::Model> {
        let typed = self.0.into_typed()?;
        Ok(Model(typed))
    }

    fn into_optimized(self) -> Result<Self::Model> {
        let typed = self.0.into_optimized()?;
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

    fn declutter(&mut self) -> Result<()> {
        self.0.declutter()
    }

    fn optimize(&mut self) -> Result<()> {
        self.0.optimize()
    }

    fn into_decluttered(mut self) -> Result<Model> {
        self.0.declutter()?;
        Ok(self)
    }

    fn into_optimized(self) -> Result<Model> {
        Ok(Model(self.0.into_optimized()?))
    }

    fn into_runnable(self) -> Result<Runnable> {
        Ok(Runnable(Arc::new(self.0.into_runnable()?)))
    }

    fn concretize_symbols(
        &mut self,
        values: impl IntoIterator<Item = (impl AsRef<str>, i64)>,
    ) -> Result<()> {
        let mut table = SymbolValues::default();
        for (k, v) in values {
            table = table.with(&self.0.symbol_table.sym(k.as_ref()), v);
        }
        self.0 = self.0.concretize_dims(&table)?;
        Ok(())
    }

    fn half(&mut self) -> Result<()> {
        self.0 = tract_nnef::tract_core::half::HalfTranslator.translate_model(&self.0)?;
        Ok(())
    }

    fn pulse(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<()> {
        let stream_sym = self.0.symbol_table.sym(name.as_ref());
        let pulse_dim = parse_tdim(&self.0.symbol_table, value.as_ref())?;
        self.0 = PulsedModel::new(&self.0, stream_sym, &pulse_dim)?.into_typed()?;
        Ok(())
    }

    fn cost_json(&self) -> Result<String> {
        let input: Option<Vec<Value>> = None;
        self.profile_json(input)
    }

    fn profile_json<I, V, E>(&self, inputs: Option<I>) -> Result<String>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error> + Debug,
    {
        let mut annotations = Annotations::from_model(&self.0)?;
        tract_libcli::profile::extract_costs(&mut annotations, &self.0)?;
        if let Some(inputs) = inputs {
            let inputs = inputs
                .into_iter()
                .map(|v| Ok(v.try_into().unwrap().0))
                .collect::<TractResult<TVec<_>>>()?;
            tract_libcli::profile::profile(
                &self.0,
                &BenchLimits::default(),
                &mut annotations,
                &inputs,
                None,
                true,
            )?;
        };
        let export = tract_libcli::export::GraphPerfInfo::from(&self.0, &annotations);
        Ok(serde_json::to_string(&export)?)
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

// RUNNABLE
pub struct Runnable(Arc<TypedRunnableModel<TypedModel>>);

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
        let state = TypedSimpleState::new(self.0.clone())?;
        Ok(State(state))
    }
}

// STATE
pub struct State(TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>);

impl StateInterface for State {
    type Value = Value;

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
}

// VALUE
pub struct Value(TValue);

impl ValueInterface for Value {
    fn from_shape_and_slice<T: TractProxyDatumType>(shape: &[usize], data: &[T]) -> Result<Value> {
        let data = tensor1(data);
        Ok(Value(data.into_shape(shape)?.into_tvalue()))
    }

    fn as_parts<T: TractProxyDatumType>(&self) -> Result<(&[usize], &[T])> {
        let shape = self.0.shape();
        let data = self.0.as_slice::<T>()?;
        Ok((shape, data))
    }

    fn view<T: TractProxyDatumType>(&self) -> Result<ndarray::ArrayViewD<T>> {
        self.0.to_array_view::<T>()
    }
}

impl<T, S, D> TryFrom<ndarray::ArrayBase<S, D>> for Value
where
    T: TractProxyDatumType,
    S: RawData<Elem = T> + Data,
    D: Dimension,
{
    type Error = anyhow::Error;
    fn try_from(view: ndarray::ArrayBase<S, D>) -> Result<Value> {
        if let Some(slice) = view.as_slice_memory_order() {
            Value::from_shape_and_slice(view.shape(), slice)
        } else {
            let slice: Vec<_> = view.iter().cloned().collect();
            Value::from_shape_and_slice(view.shape(), &slice)
        }
    }
}

impl<'a, T: Datum> TryFrom<&'a Value> for ndarray::ArrayViewD<'a, T> {
    type Error = anyhow::Error;
    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        value.0.to_array_view()
    }
}

pub struct Fact(TypedFact);

impl FactInterface for Fact {}

impl Fact {
    fn new(model: &mut Model, spec: impl ToString) -> Result<Fact> {
        let fact = tract_libcli::tensor::parse_spec(&mut model.0.symbol_table, &spec.to_string())?;
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

pub struct InferenceFact(tract_onnx::prelude::InferenceFact);

impl InferenceFactInterface for InferenceFact {}

impl InferenceFact {
    fn new(model: &mut InferenceModel, spec: impl ToString) -> Result<InferenceFact> {
        let fact = tract_libcli::tensor::parse_spec(&mut model.0.symbol_table, &spec.to_string())?;
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

as_inference_fact_impl!(InferenceModel, InferenceFact);
as_fact_impl!(Model, Fact);
