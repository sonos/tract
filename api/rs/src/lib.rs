#[cfg(target_vendor = "apple")]
extern crate tract_metal;

#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate tract_cuda;
extern crate tract_transformers;

use std::fmt::{Debug, Display};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tract_extra::WithTractExtra;
use tract_libcli::annotations::Annotations;
use tract_libcli::profile::BenchLimits;
use tract_libcli::tensor::RunTensors;
use tract_nnef::internal::Runtime as _;
use tract_nnef::prelude::{
    Framework, IntoArcTensor, IntoTValue, SymbolValues, TDim, TValue, TVec,
    Tensor as InternalTensor, TractResult, TypedFact, TypedModel, TypedSimplePlan,
};
use tract_onnx::prelude::InferenceModelExt;
use tract_onnx_opl::WithOnnx;
use tract_pulse::WithPulse;
use tract_transformers::WithTractTransformers;

use tract_api::*;

#[doc(hidden)]
pub use ::tract_api::impl_ndarray_interop_with as __impl_ndarray_interop_with;

#[doc(hidden)]
pub use ::tract_api::{Datum as __Datum, TensorInterface as __TensorInterface};

/// Generate ndarray interop for `tract::Tensor` using the caller crate's
/// own `ndarray` version.
///
/// `tract` itself has no public `ndarray` dependency: the tensor interface
/// deals only in shapes, slices, bytes and primitive datums. If your
/// application wants the ergonomics of `ndarray`, invoke this macro once
/// at the root of your crate. The macro expands in your crate's scope, so
/// the `ndarray::*` types referenced in the generated code resolve against
/// *your* `ndarray` dependency — freeing you from whichever version
/// `tract` happens to use internally.
///
/// Two traits are generated:
///
/// - `Tract`, with method `fn tract(self) -> anyhow::Result<tract::Tensor>`,
///   implemented for every `ndarray::ArrayBase<S, D>` with a `Datum`
///   element type.
/// - `Ndarray`, with methods `ndarray::<T>()`, `ndarray0::<T>()` …
///   `ndarray6::<T>()`, implemented for `tract::Tensor`. `ndarrayN`
///   returns a rank-`N` `ArrayView`; `ndarray` returns the dynamic-rank
///   `ArrayViewD`.
///
/// # Example
///
/// ```ignore
/// tract::impl_ndarray_interop!();
///
/// use ndarray::Array4;
///
/// let input: tract::Tensor = Array4::<f32>::zeros((1, 3, 224, 224)).tract()?;
/// let outputs = model.run([input])?;
/// let view = outputs[0].ndarray::<f32>()?;          // ArrayViewD<f32>
/// let v4   = outputs[0].ndarray4::<f32>()?;         // ArrayView4<f32>
/// ```
///
/// Your crate must depend on `ndarray` and `anyhow` directly for this
/// macro to compile.
#[macro_export]
macro_rules! impl_ndarray_interop {
    () => {
        $crate::__impl_ndarray_interop_with!(
            $crate::Tensor,
            $crate::__Datum,
            $crate::__TensorInterface,
        );
    };
}

pub mod prelude {
    // Concrete types
    pub use crate::{
        Dim, Fact, InferenceFact, InferenceModel, Model, Nnef, Onnx, Runnable, Runtime, State,
        Tensor, nnef, onnx, runtime_for_name,
    };

    // User-facing API types
    pub use tract_api::{
        ConcretizeSymbols, Datum, DatumType, FloatPrecision, Pulse, TransformSpec, tensor,
    };

    // Traits needed for method resolution — hidden from namespace
    pub use tract_api::{
        DimInterface as _, FactInterface as _, InferenceFactInterface as _,
        InferenceModelInterface as _, ModelInterface as _, NnefInterface as _, OnnxInterface as _,
        RunnableInterface as _, RuntimeInterface as _, StateInterface as _, TensorInterface as _,
    };
}

/// Creates an instance of an NNEF framework and parser that can be used to load and dump NNEF models.
pub fn nnef() -> Result<Nnef> {
    Ok(Nnef(tract_nnef::nnef()))
}

pub fn onnx() -> Result<Onnx> {
    Ok(Onnx(tract_onnx::onnx()))
}

pub fn runtime_for_name(name: &str) -> Result<Runtime> {
    if let Some(rt) = tract_onnx::tract_core::runtime::runtime_for_name(name)? {
        Ok(Runtime(rt))
    } else {
        anyhow::bail!("Unknown runtime {name} (not compiled in tract?)")
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

    fn load_buffer(&self, data: &[u8]) -> Result<Self::Model> {
        let m = self.0.model_for_read(&mut Cursor::new(data))?.into_decluttered()?;
        Ok(Model(m))
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

impl Debug for Onnx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Onnx")
    }
}

impl OnnxInterface for Onnx {
    type InferenceModel = InferenceModel;
    fn load(&self, path: impl AsRef<Path>) -> Result<Self::InferenceModel> {
        Ok(InferenceModel(self.0.model_for_path(path)?))
    }

    fn load_buffer(&self, data: &[u8]) -> Result<Self::InferenceModel> {
        let m = self.0.model_for_read(&mut Cursor::new(data))?;
        Ok(InferenceModel(m))
    }
}

pub struct InferenceModel(tract_onnx::prelude::InferenceModel);

impl Debug for InferenceModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InferenceModel")
    }
}

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

    fn into_model(self) -> Result<Self::Model> {
        let typed = self.0.into_typed()?.into_decluttered()?;
        Ok(Model(typed))
    }
}

// MODEL
#[derive(Debug, Clone)]
pub struct Model(TypedModel);

impl ModelInterface for Model {
    type Fact = Fact;
    type Runnable = Runnable;
    type Tensor = Tensor;

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

    fn transform(&mut self, spec: impl Into<TransformSpec>) -> Result<()> {
        let transform = spec.into().to_transform_string();
        let transform_obj = if transform.trim_start().starts_with('{') {
            // JSON input: parse, extract name, deserialize params
            let v: serde_json::Value = serde_json::from_str(&transform)?;
            let obj = v.as_object().context("expected JSON object")?;
            let name = obj
                .get("name")
                .and_then(|v| v.as_str())
                .context("missing 'name' field")?
                .to_string();
            let mut params = v.clone();
            params.as_object_mut().unwrap().remove("name");
            let mut erased = <dyn erased_serde::Deserializer>::erase(params);
            tract_onnx::tract_core::transform::get_transform_with_params(&name, &mut erased)?
                .with_context(|| format!("transform `{name}' could not be found"))?
        } else {
            // Plain name (no params)
            tract_onnx::tract_core::transform::get_transform(&transform)?
                .with_context(|| format!("transform `{transform}' could not be found"))?
        };
        transform_obj.transform(&mut self.0)?;
        self.0.declutter()
    }

    fn parse_fact(&self, spec: &str) -> Result<Fact> {
        let f = spec.as_fact(self)?;
        Ok(Fact(f.0.clone()))
    }

    fn property_keys(&self) -> Result<Vec<String>> {
        Ok(self.0.properties.keys().cloned().collect())
    }

    fn property(&self, name: impl AsRef<str>) -> Result<Tensor> {
        let name = name.as_ref();
        self.0
            .properties
            .get(name)
            .with_context(|| format!("no property for name {name}"))
            .map(|t| Tensor(t.clone()))
    }
}

// RUNTIME
pub struct Runtime(&'static dyn tract_nnef::internal::Runtime);

impl Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Runtime({:?})", self.0.name())
    }
}

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
#[derive(Debug, Clone)]
pub struct Runnable(Arc<dyn tract_nnef::internal::Runnable>);

impl RunnableInterface for Runnable {
    type Tensor = Tensor;
    type State = State;
    type Fact = Fact;

    fn run(&self, inputs: impl IntoInputs<Tensor>) -> Result<Vec<Tensor>> {
        StateInterface::run(&mut self.spawn_state()?, inputs.into_inputs()?)
    }

    fn spawn_state(&self) -> Result<State> {
        let state = self.0.spawn()?;
        Ok(State(Some(state.freeze_into())))
    }

    fn input_count(&self) -> Result<usize> {
        Ok(self.0.input_count())
    }

    fn output_count(&self) -> Result<usize> {
        Ok(self.0.output_count())
    }

    fn input_fact(&self, id: usize) -> Result<Fact> {
        Ok(Fact(self.0.input_fact(id)?.clone()))
    }

    fn output_fact(&self, id: usize) -> Result<Fact> {
        Ok(Fact(self.0.output_fact(id)?.clone()))
    }

    fn property_keys(&self) -> Result<Vec<String>> {
        Ok(self.0.properties().keys().cloned().collect())
    }

    fn property(&self, name: impl AsRef<str>) -> Result<Tensor> {
        let name = name.as_ref();
        self.0
            .properties()
            .get(name)
            .with_context(|| format!("no property for name {name}"))
            .map(|t| Tensor(t.clone()))
    }

    fn cost_json(&self) -> Result<String> {
        let input: Option<Vec<Tensor>> = None;
        self.profile_json(input)
    }

    fn profile_json<I, IV, IE>(&self, inputs: Option<I>) -> Result<String>
    where
        I: IntoIterator<Item = IV>,
        IV: TryInto<Self::Tensor, Error = IE>,
        IE: Into<anyhow::Error> + Debug,
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
                .map(|v| Ok(v.try_into().unwrap().0.into_tvalue()))
                .collect::<TractResult<TVec<_>>>()?;

            tract_libcli::profile::profile(
                &self.0,
                &BenchLimits::default(),
                &mut annotations,
                &RunTensors { sources: vec![inputs] },
                None,
                true,
            )?;
        };
        let export = tract_libcli::export::GraphPerfInfo::from(model, &annotations);
        Ok(serde_json::to_string(&export)?)
    }
}

// STATE
pub struct State(Option<Box<dyn tract_nnef::internal::FrozenState>>);

impl Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State")
    }
}

impl Clone for State {
    fn clone(&self) -> Self {
        State(self.0.as_ref().map(|s| tract_nnef::tract_core::dyn_clone::clone_box(&**s)))
    }
}

// Safety: FrozenState is Send
unsafe impl Send for State {}

impl StateInterface for State {
    type Fact = Fact;
    type Tensor = Tensor;

    fn input_count(&self) -> Result<usize> {
        Ok(self.0.as_ref().context("State has been invalidated")?.input_count())
    }

    fn output_count(&self) -> Result<usize> {
        Ok(self.0.as_ref().context("State has been invalidated")?.output_count())
    }

    fn run(&mut self, inputs: impl IntoInputs<Tensor>) -> Result<Vec<Tensor>> {
        let inputs: TVec<TValue> =
            inputs.into_inputs()?.into_iter().map(|v| v.0.into_tvalue()).collect();
        let frozen = self.0.take().context("State has been invalidated")?;
        let mut state = frozen.unfreeze();
        let outputs = state.run(inputs)?;
        self.0 = Some(state.freeze_into());
        Ok(outputs.into_iter().map(|t| Tensor(t.into_arc_tensor())).collect())
    }
}

// TENSOR
#[derive(Clone, Debug)]
pub struct Tensor(Arc<InternalTensor>);

impl TensorInterface for Tensor {
    fn datum_type(&self) -> Result<DatumType> {
        from_internal_dt(self.0.datum_type())
    }

    fn from_bytes(dt: DatumType, shape: &[usize], data: &[u8]) -> Result<Self> {
        let dt = to_internal_dt(dt);
        let len = shape.iter().product::<usize>() * dt.size_of();
        anyhow::ensure!(len == data.len());
        let tensor = unsafe { InternalTensor::from_raw_dt(dt, shape, data)? };
        Ok(Tensor(tensor.into_arc_tensor()))
    }

    fn as_bytes(&self) -> Result<(DatumType, &[usize], &[u8])> {
        let dt = from_internal_dt(self.0.datum_type())?;
        Ok((dt, self.0.shape(), self.0.try_as_plain()?.as_bytes()))
    }

    fn convert_to(&self, to: DatumType) -> Result<Self> {
        let to = to_internal_dt(to);
        if self.0.datum_type() == to {
            Ok(self.clone())
        } else {
            Ok(Tensor(self.0.cast_to_dt(to)?.into_owned().into_arc_tensor()))
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let Ok((me_dt, me_shape, me_data)) = self.as_bytes() else { return false };
        let Ok((other_dt, other_shape, other_data)) = other.as_bytes() else { return false };
        me_dt == other_dt && me_shape == other_shape && me_data == other_data
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.dump(false).map_err(|_| std::fmt::Error)?)
    }
}

#[derive(Clone, Debug)]
pub struct Fact(TypedFact);

impl FactInterface for Fact {
    type Dim = Dim;

    fn datum_type(&self) -> Result<DatumType> {
        from_internal_dt(self.0.datum_type)
    }

    fn rank(&self) -> Result<usize> {
        Ok(self.0.rank())
    }

    fn dim(&self, axis: usize) -> Result<Self::Dim> {
        anyhow::ensure!(axis < self.0.rank());
        Ok(Dim(self.0.shape[axis].clone()))
    }
}

impl Fact {
    fn new(model: &Model, spec: impl ToString) -> Result<Fact> {
        let fact = tract_libcli::tensor::parse_spec(&model.0.symbols, &spec.to_string())?;
        let fact = tract_onnx::prelude::Fact::to_typed_fact(&fact)?.into_owned();
        Ok(Fact(fact))
    }
}

impl Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, dim) in self.0.shape.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{dim}")?;
        }
        let dt = format!("{:?}", self.0.datum_type).to_lowercase();
        if self.0.rank() > 0 {
            write!(f, ",")?;
        }
        write!(f, "{dt}")
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
    fn new(model: &InferenceModel, spec: impl ToString) -> Result<InferenceFact> {
        let fact = tract_libcli::tensor::parse_spec(&model.0.symbols, &spec.to_string())?;
        Ok(InferenceFact(fact))
    }
}

impl Display for InferenceFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.0.format_dt_shape();
        // Lowercase the trailing datum type token (e.g. "F32" → "f32")
        // to match the format accepted by the parser.
        if let Some(pos) = s.rfind(',') {
            let (dims, dt) = s.split_at(pos + 1);
            write!(f, "{dims}{}", dt.to_lowercase())
        } else {
            // Scalar or unknown: the entire string is the dtype
            write!(f, "{}", s.to_lowercase())
        }
    }
}

as_inference_fact_impl!(InferenceModel, InferenceFact);
as_fact_impl!(Model, Fact);

#[derive(Clone, Debug)]
pub struct Dim(TDim);

impl DimInterface for Dim {
    fn eval(&self, values: impl IntoIterator<Item = (impl AsRef<str>, i64)>) -> Result<Dim> {
        if let Some(scope) = self.0.find_scope() {
            let mut table = SymbolValues::default();
            for (k, v) in values {
                table = table.with(&scope.sym(k.as_ref()), v);
            }
            let result = self.0.eval(&table);
            Ok(Dim(result))
        } else {
            Ok(self.clone())
        }
    }

    fn to_int64(&self) -> Result<i64> {
        self.0.to_i64()
    }
}

impl Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/*
#[inline(always)]
fn to_datum_type<T: TractProxyDatumType>() -> Result<tract_nnef::prelude::DatumType> {
macro_rules! dt { ($($t:ty),*) => { $(if TypeId::of::<T>() == TypeId::of::<$t>() { return Ok(<$t>::datum_type()); })* }}
dt!(f32, f16, f64, i64, i32, i16, i8, bool, u64, u32, u16, u8);
anyhow::bail!("Unsupported type {}", std::any::type_name::<T>())
}
*/

fn to_internal_dt(it: DatumType) -> tract_nnef::prelude::DatumType {
    type Api = DatumType;
    type Internal = tract_nnef::prelude::DatumType;
    match it {
        Api::Bool => Internal::Bool,
        Api::U8 => Internal::U8,
        Api::U16 => Internal::U16,
        Api::U32 => Internal::U32,
        Api::U64 => Internal::U64,
        Api::I8 => Internal::I8,
        Api::I16 => Internal::I16,
        Api::I32 => Internal::I32,
        Api::I64 => Internal::I64,
        Api::F16 => Internal::F16,
        Api::F32 => Internal::F32,
        Api::F64 => Internal::F64,
        #[cfg(feature = "complex")]
        Api::ComplexI16 => Internal::ComplexI16,
        #[cfg(feature = "complex")]
        Api::ComplexI32 => Internal::ComplexI32,
        #[cfg(feature = "complex")]
        Api::ComplexI64 => Internal::ComplexI64,
        #[cfg(feature = "complex")]
        Api::ComplexF16 => Internal::ComplexF16,
        #[cfg(feature = "complex")]
        Api::ComplexF32 => Internal::ComplexF32,
        #[cfg(feature = "complex")]
        Api::ComplexF64 => Internal::ComplexF64,
    }
}

fn from_internal_dt(it: tract_nnef::prelude::DatumType) -> Result<DatumType> {
    type Api = DatumType;
    type Internal = tract_nnef::prelude::DatumType;
    Ok(match it {
        Internal::Bool => Api::Bool,
        Internal::U8 => Api::U8,
        Internal::U16 => Api::U16,
        Internal::U32 => Api::U32,
        Internal::U64 => Api::U64,
        Internal::I8 => Api::I8,
        Internal::I16 => Api::I16,
        Internal::I32 => Api::I32,
        Internal::I64 => Api::I64,
        Internal::F16 => Api::F16,
        Internal::F32 => Api::F32,
        Internal::F64 => Api::F64,
        #[cfg(feature = "complex")]
        Internal::ComplexI16 => Api::ComplexI16,
        #[cfg(feature = "complex")]
        Internal::ComplexI32 => Api::ComplexI32,
        #[cfg(feature = "complex")]
        Internal::ComplexI64 => Api::ComplexI64,
        #[cfg(feature = "complex")]
        Internal::ComplexF16 => Api::ComplexF16,
        #[cfg(feature = "complex")]
        Internal::ComplexF32 => Api::ComplexF32,
        #[cfg(feature = "complex")]
        Internal::ComplexF64 => Api::ComplexF64,
        _ => {
            anyhow::bail!("Unsupported DatumType in the public API {:?}", it)
        }
    })
}
