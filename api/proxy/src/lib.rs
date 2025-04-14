use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::{null, null_mut};

use tract_api::*;
use tract_proxy_sys as sys;

use anyhow::{Context, Result};
use ndarray::*;

macro_rules! check {
    ($expr:expr) => {
        unsafe {
            if $expr == sys::TRACT_RESULT_TRACT_RESULT_KO {
                let buf = CStr::from_ptr(sys::tract_get_last_error());
                Err(anyhow::anyhow!(buf.to_string_lossy().to_string()))
            } else {
                Ok(())
            }
        }
    };
}

macro_rules! wrapper {
    ($new_type:ident, $c_type:ident, $dest:ident $(, $typ:ty )*) => {
        #[derive(Debug, Clone)]
        pub struct $new_type(*mut sys::$c_type $(, $typ)*);

        impl Drop for $new_type {
            fn drop(&mut self) {
                unsafe {
                    sys::$dest(&mut self.0);
                }
            }
        }
    };
}

pub fn nnef() -> Result<Nnef> {
    let mut nnef = null_mut();
    check!(sys::tract_nnef_create(&mut nnef))?;
    Ok(Nnef(nnef))
}

pub fn onnx() -> Result<Onnx> {
    let mut onnx = null_mut();
    check!(sys::tract_onnx_create(&mut onnx))?;
    Ok(Onnx(onnx))
}

pub fn version() -> &'static str {
    unsafe { CStr::from_ptr(sys::tract_version()).to_str().unwrap() }
}

wrapper!(Nnef, TractNnef, tract_nnef_destroy);
impl NnefInterface for Nnef {
    type Model = Model;
    fn model_for_path(&self, path: impl AsRef<Path>) -> Result<Model> {
        let path = path.as_ref();
        let path = CString::new(
            path.to_str().with_context(|| format!("Failed to re-encode {path:?} to uff-8"))?,
        )?;
        let mut model = null_mut();
        check!(sys::tract_nnef_model_for_path(self.0, path.as_ptr(), &mut model))?;
        Ok(Model(model))
    }

    fn transform_model(&self, model: &mut Self::Model, transform_spec: &str) -> Result<()> {
        let t = CString::new(transform_spec)?;
        check!(sys::tract_nnef_transform_model(self.0, model.0, t.as_ptr()))
    }

    fn enable_tract_core(&mut self) -> Result<()> {
        check!(sys::tract_nnef_enable_tract_core(self.0))
    }

    fn enable_tract_extra(&mut self) -> Result<()> {
        check!(sys::tract_nnef_enable_tract_extra(self.0))
    }

    fn enable_tract_transformers(&mut self) -> Result<()> {
        check!(sys::tract_nnef_enable_tract_transformers(self.0))
    }

    fn enable_onnx(&mut self) -> Result<()> {
        check!(sys::tract_nnef_enable_onnx(self.0))
    }

    fn enable_pulse(&mut self) -> Result<()> {
        check!(sys::tract_nnef_enable_pulse(self.0))
    }

    fn enable_extended_identifier_syntax(&mut self) -> Result<()> {
        check!(sys::tract_nnef_enable_extended_identifier_syntax(self.0))
    }

    fn write_model_to_dir(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let path = path.as_ref();
        let path = CString::new(
            path.to_str().with_context(|| format!("Failed to re-encode {path:?} to uff-8"))?,
        )?;
        check!(sys::tract_nnef_write_model_to_dir(self.0, path.as_ptr(), model.0))?;
        Ok(())
    }

    fn write_model_to_tar(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let path = path.as_ref();
        let path = CString::new(
            path.to_str().with_context(|| format!("Failed to re-encode {path:?} to uff-8"))?,
        )?;
        check!(sys::tract_nnef_write_model_to_tar(self.0, path.as_ptr(), model.0))?;
        Ok(())
    }

    fn write_model_to_tar_gz(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let path = path.as_ref();
        let path = CString::new(
            path.to_str().with_context(|| format!("Failed to re-encode {path:?} to uff-8"))?,
        )?;
        check!(sys::tract_nnef_write_model_to_tar_gz(self.0, path.as_ptr(), model.0))?;
        Ok(())
    }
}

// ONNX
wrapper!(Onnx, TractOnnx, tract_onnx_destroy);

impl OnnxInterface for Onnx {
    type InferenceModel = InferenceModel;
    fn model_for_path(&self, path: impl AsRef<Path>) -> Result<InferenceModel> {
        let path = path.as_ref();
        let path = CString::new(
            path.to_str().with_context(|| format!("Failed to re-encode {path:?} to uff-8"))?,
        )?;
        let mut model = null_mut();
        check!(sys::tract_onnx_model_for_path(self.0, path.as_ptr(), &mut model))?;
        Ok(InferenceModel(model))
    }
}

// INFERENCE MODEL
wrapper!(InferenceModel, TractInferenceModel, tract_inference_model_destroy);
impl InferenceModelInterface for InferenceModel {
    type Model = Model;
    type InferenceFact = InferenceFact;
    fn set_output_names(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<()> {
        let c_strings: Vec<CString> =
            outputs.into_iter().map(|a| Ok(CString::new(a.as_ref())?)).collect::<Result<_>>()?;
        let ptrs: Vec<_> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
        check!(sys::tract_inference_model_set_output_names(
            self.0,
            c_strings.len(),
            ptrs.as_ptr()
        ))?;
        Ok(())
    }

    fn input_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_inference_model_input_count(self.0, &mut count))?;
        Ok(count)
    }

    fn output_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_inference_model_output_count(self.0, &mut count))?;
        Ok(count)
    }

    fn input_name(&self, id: usize) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_input_name(self.0, id, &mut ptr))?;
        unsafe {
            let ret = CStr::from_ptr(ptr).to_str()?.to_owned();
            sys::tract_free_cstring(ptr);
            Ok(ret)
        }
    }

    fn output_name(&self, id: usize) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_output_name(self.0, id, &mut ptr))?;
        unsafe {
            let ret = CStr::from_ptr(ptr).to_str()?.to_owned();
            sys::tract_free_cstring(ptr);
            Ok(ret)
        }
    }

    fn input_fact(&self, id: usize) -> Result<InferenceFact> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_input_fact(self.0, id, &mut ptr))?;
        Ok(InferenceFact(ptr))
    }

    fn set_input_fact(
        &mut self,
        id: usize,
        fact: impl AsFact<Self, Self::InferenceFact>,
    ) -> Result<()> {
        let fact = fact.as_fact(self)?;
        check!(sys::tract_inference_model_set_input_fact(self.0, id, fact.0))?;
        Ok(())
    }

    fn output_fact(&self, id: usize) -> Result<InferenceFact> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_output_fact(self.0, id, &mut ptr))?;
        Ok(InferenceFact(ptr))
    }

    fn set_output_fact(
        &mut self,
        id: usize,
        fact: impl AsFact<InferenceModel, InferenceFact>,
    ) -> Result<()> {
        let fact = fact.as_fact(self)?;
        check!(sys::tract_inference_model_set_output_fact(self.0, id, fact.0))?;
        Ok(())
    }

    fn analyse(&mut self) -> Result<()> {
        check!(sys::tract_inference_model_analyse(self.0))?;
        Ok(())
    }

    fn into_typed(mut self) -> Result<Self::Model> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_into_typed(&mut self.0, &mut ptr))?;
        Ok(Model(ptr))
    }

    fn into_optimized(mut self) -> Result<Self::Model> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_into_optimized(&mut self.0, &mut ptr))?;
        Ok(Model(ptr))
    }
}

// MODEL
wrapper!(Model, TractModel, tract_model_destroy);

impl ModelInterface for Model {
    type Fact = Fact;
    type Value = Value;
    type Runnable = Runnable;
    fn input_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_model_input_count(self.0, &mut count))?;
        Ok(count)
    }

    fn output_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_model_output_count(self.0, &mut count))?;
        Ok(count)
    }

    fn input_name(&self, id: usize) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_model_input_name(self.0, id, &mut ptr))?;
        unsafe {
            let ret = CStr::from_ptr(ptr).to_str()?.to_owned();
            sys::tract_free_cstring(ptr);
            Ok(ret)
        }
    }

    fn output_name(&self, id: usize) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_model_output_name(self.0, id, &mut ptr))?;
        unsafe {
            let ret = CStr::from_ptr(ptr).to_str()?.to_owned();
            sys::tract_free_cstring(ptr);
            Ok(ret)
        }
    }

    fn set_output_names(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<()> {
        let c_strings: Vec<CString> =
            outputs.into_iter().map(|a| Ok(CString::new(a.as_ref())?)).collect::<Result<_>>()?;
        let ptrs: Vec<_> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
        check!(sys::tract_model_set_output_names(self.0, c_strings.len(), ptrs.as_ptr()))?;
        Ok(())
    }

    fn input_fact(&self, id: usize) -> Result<Fact> {
        let mut ptr = null_mut();
        check!(sys::tract_model_input_fact(self.0, id, &mut ptr))?;
        Ok(Fact(ptr))
    }

    fn output_fact(&self, id: usize) -> Result<Fact> {
        let mut ptr = null_mut();
        check!(sys::tract_model_output_fact(self.0, id, &mut ptr))?;
        Ok(Fact(ptr))
    }

    fn declutter(&mut self) -> Result<()> {
        check!(sys::tract_model_declutter(self.0))?;
        Ok(())
    }

    fn optimize(&mut self) -> Result<()> {
        check!(sys::tract_model_optimize(self.0))?;
        Ok(())
    }

    fn into_decluttered(self) -> Result<Model> {
        check!(sys::tract_model_declutter(self.0))?;
        Ok(self)
    }

    fn into_optimized(self) -> Result<Model> {
        check!(sys::tract_model_optimize(self.0))?;
        Ok(self)
    }

    fn into_runnable(self) -> Result<Runnable> {
        let mut model = self;
        let mut runnable = null_mut();
        check!(sys::tract_model_into_runnable(&mut model.0, &mut runnable))?;
        Ok(Runnable(runnable))
    }

    fn concretize_symbols(
        &mut self,
        values: impl IntoIterator<Item = (impl AsRef<str>, i64)>,
    ) -> Result<()> {
        let (names, values): (Vec<_>, Vec<_>) = values.into_iter().unzip();
        let c_strings: Vec<CString> =
            names.into_iter().map(|a| Ok(CString::new(a.as_ref())?)).collect::<Result<_>>()?;
        let ptrs: Vec<_> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
        check!(sys::tract_model_concretize_symbols(
            self.0,
            ptrs.len(),
            ptrs.as_ptr(),
            values.as_ptr()
        ))?;
        Ok(())
    }

    fn transform(&mut self, transform: &str) -> Result<()> {
        let t = CString::new(transform)?;
        check!(sys::tract_model_transform(self.0, t.as_ptr()))?;
        Ok(())
    }

    fn pulse(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<()> {
        let name = CString::new(name.as_ref())?;
        let value = CString::new(value.as_ref())?;
        check!(sys::tract_model_pulse_simple(&mut self.0, name.as_ptr(), value.as_ptr()))?;
        Ok(())
    }

    fn cost_json(&self) -> Result<String> {
        let input: Option<Vec<Value>> = None;
        self.profile_json(input)
    }

    fn profile_json<I, V, E>(&self, inputs: Option<I>) -> Result<String>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Value, Error = E>,
        E: Into<anyhow::Error>,
    {
        let inputs = if let Some(inputs) = inputs {
            let inputs = inputs
                .into_iter()
                .map(|i| i.try_into().map_err(|e| e.into()))
                .collect::<Result<Vec<Value>>>()?;
            anyhow::ensure!(self.input_count()? == inputs.len());
            Some(inputs)
        } else {
            None
        };
        let mut iptrs: Option<Vec<*mut sys::TractValue>> =
            inputs.as_ref().map(|is| is.iter().map(|v| v.0).collect());
        let mut json: *mut i8 = null_mut();
        let values = iptrs.as_mut().map(|it| it.as_mut_ptr()).unwrap_or(null_mut());
        check!(sys::tract_model_profile_json(self.0, values, &mut json))?;
        anyhow::ensure!(!json.is_null());
        unsafe {
            let s = CStr::from_ptr(json).to_owned();
            sys::tract_free_cstring(json);
            Ok(s.to_str()?.to_owned())
        }
    }

    fn property_keys(&self) -> Result<Vec<String>> {
        let mut len = 0;
        check!(sys::tract_model_property_count(self.0, &mut len))?;
        let mut keys = vec![null_mut(); len];
        check!(sys::tract_model_property_names(self.0, keys.as_mut_ptr()))?;
        unsafe {
            keys.into_iter()
                .map(|pc| {
                    let s = CStr::from_ptr(pc).to_str()?.to_owned();
                    sys::tract_free_cstring(pc);
                    Ok(s)
                })
                .collect()
        }
    }

    fn property(&self, name: impl AsRef<str>) -> Result<Value> {
        let mut v = null_mut();
        let name = CString::new(name.as_ref())?;
        check!(sys::tract_model_property(self.0, name.as_ptr(), &mut v))?;
        Ok(Value(v))
    }
}

// RUNNABLE
wrapper!(Runnable, TractRunnable, tract_runnable_release);

impl RunnableInterface for Runnable {
    type Value = Value;
    type State = State;

    fn run<I, V, E>(&self, inputs: I) -> Result<Vec<Value>>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Value, Error = E>,
        E: Into<anyhow::Error>,
    {
        self.spawn_state()?.run(inputs)
    }

    fn spawn_state(&self) -> Result<State> {
        let mut state = null_mut();
        check!(sys::tract_runnable_spawn_state(self.0, &mut state))?;
        Ok(State(state))
    }

    fn input_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_runnable_input_count(self.0, &mut count))?;
        Ok(count)
    }

    fn output_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_runnable_output_count(self.0, &mut count))?;
        Ok(count)
    }
}

// STATE
wrapper!(State, TractState, tract_state_destroy);

impl StateInterface for State {
    type Value = Value;
    fn run<I, V, E>(&mut self, inputs: I) -> Result<Vec<Value>>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Value, Error = E>,
        E: Into<anyhow::Error>,
    {
        let inputs = inputs
            .into_iter()
            .map(|i| i.try_into().map_err(|e| e.into()))
            .collect::<Result<Vec<Value>>>()?;
        let mut outputs = vec![null_mut(); self.output_count()?];
        let mut inputs: Vec<_> = inputs.iter().map(|v| v.0).collect();
        check!(sys::tract_state_run(self.0, inputs.as_mut_ptr(), outputs.as_mut_ptr()))?;
        let outputs = outputs.into_iter().map(Value).collect();
        Ok(outputs)
    }

    fn input_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_state_input_count(self.0, &mut count))?;
        Ok(count)
    }

    fn output_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_state_output_count(self.0, &mut count))?;
        Ok(count)
    }
}

// VALUE
wrapper!(Value, TractValue, tract_value_destroy);

impl ValueInterface for Value {
    fn from_bytes(dt: DatumType, shape: &[usize], data: &[u8]) -> Result<Self> {
        anyhow::ensure!(data.len() == shape.iter().product::<usize>() * dt.size_of());
        let mut value = null_mut();
        check!(sys::tract_value_from_bytes(
            dt as _,
            shape.len(),
            shape.as_ptr(),
            data.as_ptr() as _,
            &mut value
        ))?;
        Ok(Value(value))
    }

    fn as_bytes(&self) -> Result<(DatumType, &[usize], &[u8])> {
        let mut rank = 0;
        let mut dt = sys::DatumType_TRACT_DATUM_TYPE_BOOL as _;
        let mut shape = null();
        let mut data = null();
        check!(sys::tract_value_as_bytes(self.0, &mut dt, &mut rank, &mut shape, &mut data))?;
        unsafe {
            let dt: DatumType = std::mem::transmute(dt);
            let shape = std::slice::from_raw_parts(shape, rank);
            let len: usize = shape.iter().product();
            let data = std::slice::from_raw_parts(data as *const u8, len * dt.size_of());
            Ok((dt, shape, data))
        }
    }
}

value_from_to_ndarray!();

// FACT
wrapper!(Fact, TractFact, tract_fact_destroy);

impl Fact {
    fn new(model: &mut Model, spec: impl ToString) -> Result<Fact> {
        let cstr = CString::new(spec.to_string())?;
        let mut fact = null_mut();
        check!(sys::tract_fact_parse(model.0, cstr.as_ptr(), &mut fact))?;
        Ok(Fact(fact))
    }

    fn dump(&self) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_fact_dump(self.0, &mut ptr))?;
        unsafe {
            let s = CStr::from_ptr(ptr).to_owned();
            sys::tract_free_cstring(ptr);
            Ok(s.to_str()?.to_owned())
        }
    }
}

impl FactInterface for Fact {}

impl std::fmt::Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dump() {
            Ok(s) => f.write_str(&s),
            Err(_) => Err(std::fmt::Error),
        }
    }
}

// INFERENCE FACT
wrapper!(InferenceFact, TractInferenceFact, tract_inference_fact_destroy);

impl InferenceFact {
    fn new(model: &mut InferenceModel, spec: impl ToString) -> Result<InferenceFact> {
        let cstr = CString::new(spec.to_string())?;
        let mut fact = null_mut();
        check!(sys::tract_inference_fact_parse(model.0, cstr.as_ptr(), &mut fact))?;
        Ok(InferenceFact(fact))
    }

    fn dump(&self) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_fact_dump(self.0, &mut ptr))?;
        unsafe {
            let s = CStr::from_ptr(ptr).to_owned();
            sys::tract_free_cstring(ptr);
            Ok(s.to_str()?.to_owned())
        }
    }
}

impl InferenceFactInterface for InferenceFact {
    fn empty() -> Result<InferenceFact> {
        let mut fact = null_mut();
        check!(sys::tract_inference_fact_empty(&mut fact))?;
        Ok(InferenceFact(fact))
    }
}

impl Default for InferenceFact {
    fn default() -> Self {
        Self::empty().unwrap()
    }
}

impl std::fmt::Display for InferenceFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dump() {
            Ok(s) => f.write_str(&s),
            Err(_) => Err(std::fmt::Error),
        }
    }
}

as_inference_fact_impl!(InferenceModel, InferenceFact);
as_fact_impl!(Model, Fact);
