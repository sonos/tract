use std::ffi::{CStr, CString};
use std::fmt::Display;
use std::os::unix::prelude::OsStrExt;
use std::path::Path;
use std::ptr::{null, null_mut};

use boow::Bow;
use ndarray::{Dimension, RawData, Data};
use tract_rs_sys as sys;

use anyhow::Result;

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

pub fn version() -> &'static str {
    unsafe { CStr::from_ptr(sys::tract_version()).to_str().unwrap() }
}

// NNEF
wrapper!(Nnef, TractNnef, tract_nnef_destroy);

pub fn nnef() -> Result<Nnef> {
    Nnef::new()
}

impl Nnef {
    pub fn new() -> Result<Nnef> {
        let mut nnef = null_mut();
        check!(sys::tract_nnef_create(&mut nnef))?;
        Ok(Nnef(nnef))
    }

    pub fn model_for_path(&self, path: impl AsRef<Path>) -> Result<Model> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes().to_vec())?;
        let mut model = null_mut();
        check!(sys::tract_nnef_model_for_path(self.0, path.as_ptr(), &mut model))?;
        Ok(Model(model))
    }

    pub fn with_tract_core(self) -> Result<Nnef> {
        check!(sys::tract_nnef_enable_tract_core(self.0))?;
        Ok(self)
    }

    pub fn with_onnx(self) -> Result<Nnef> {
        check!(sys::tract_nnef_enable_onnx(self.0))?;
        Ok(self)
    }

    pub fn with_extended_identifier_syntax(self) -> Result<Nnef> {
        check!(sys::tract_nnef_allow_extended_identifier_syntax(self.0, true))?;
        Ok(self)
    }

    pub fn write_model_to_dir(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes().to_vec())?;
        check!(sys::tract_nnef_write_model_to_dir(self.0, path.as_ptr(), model.0))?;
        Ok(())
    }

    pub fn write_model_to_tar(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes().to_vec())?;
        check!(sys::tract_nnef_write_model_to_tar(self.0, path.as_ptr(), model.0))?;
        Ok(())
    }

    pub fn write_model_to_tar_gz(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes().to_vec())?;
        check!(sys::tract_nnef_write_model_to_tar_gz(self.0, path.as_ptr(), model.0))?;
        Ok(())
    }
}

// ONNX
wrapper!(Onnx, TractOnnx, tract_onnx_destroy);

pub fn onnx() -> Result<Onnx> {
    Onnx::new()
}

impl Onnx {
    pub fn new() -> Result<Onnx> {
        let mut onnx = null_mut();
        check!(sys::tract_onnx_create(&mut onnx))?;
        Ok(Onnx(onnx))
    }

    pub fn model_for_path(&self, path: impl AsRef<Path>) -> Result<InferenceModel> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes().to_vec())?;
        let mut model = null_mut();
        check!(sys::tract_onnx_model_for_path(self.0, path.as_ptr(), &mut model))?;
        Ok(InferenceModel(model))
    }
}

// INFERENCE MODEL
wrapper!(InferenceModel, TractInferenceModel, tract_inference_model_destroy);

impl InferenceModel {
    pub fn set_output_names(&mut self, outputs: impl IntoIterator<Item = impl AsRef<str>>) -> Result<()> {
        let c_strings:Vec<CString> = outputs.into_iter().map(|a| Ok(CString::new(a.as_ref())?)).collect::<Result<_>>()?;
        let ptrs:Vec<_> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
        check!(sys::tract_inference_model_set_output_names(self.0, c_strings.len(), ptrs.as_ptr()))?;
        Ok(())
    }

    pub fn input_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_inference_model_nbio(self.0, &mut count, null_mut()))?;
        Ok(count)
    }

    pub fn output_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_inference_model_nbio(self.0, null_mut(), &mut count))?;
        Ok(count)
    }

    pub fn input_name(&self, id: usize) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_input_name(self.0, id, &mut ptr))?;
        unsafe { Ok(CStr::from_ptr(ptr).to_str()?.to_owned()) }
    }

    pub fn output_name(&self, id: usize) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_output_name(self.0, id, &mut ptr))?;
        unsafe { Ok(CStr::from_ptr(ptr).to_str()?.to_owned()) }
    }

    pub fn input_fact(&self, id: usize) -> Result<InferenceFact> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_input_fact(self.0, id, &mut ptr))?;
        Ok(InferenceFact(ptr))
    }

    pub fn set_input_fact(
        &mut self,
        id: usize,
        fact: impl AsFact<InferenceModel, InferenceFact>,
    ) -> Result<()> {
        let fact = fact.as_fact(self)?;
        check!(sys::tract_inference_model_set_input_fact(self.0, id, fact.0))?;
        Ok(())
    }

    pub fn output_fact(&self, id: usize) -> Result<InferenceFact> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_output_fact(self.0, id, &mut ptr))?;
        Ok(InferenceFact(ptr))
    }

    pub fn set_output_fact(
        &mut self,
        id: usize,
        fact: impl AsFact<InferenceModel, InferenceFact>,
    ) -> Result<()> {
        let fact = fact.as_fact(self)?;
        check!(sys::tract_inference_model_set_output_fact(self.0, id, fact.0))?;
        Ok(())
    }

    pub fn analyse(&mut self) -> Result<()> {
        check!(sys::tract_inference_model_analyse(self.0, true))?;
        Ok(())
    }

    pub fn into_typed(mut self) -> Result<Model> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_into_typed(&mut self.0, &mut ptr))?;
        Ok(Model(ptr))
    }

    pub fn into_optimized(mut self) -> Result<Model> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_model_into_optimized(&mut self.0, &mut ptr))?;
        Ok(Model(ptr))
    }
}

// MODEL
wrapper!(Model, TractModel, tract_model_destroy);

impl Model {
    pub fn input_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_model_nbio(self.0, &mut count, null_mut()))?;
        Ok(count)
    }

    pub fn output_count(&self) -> Result<usize> {
        let mut count = 0;
        check!(sys::tract_model_nbio(self.0, null_mut(), &mut count))?;
        Ok(count)
    }

    pub fn input_name(&self, id: usize) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_model_input_name(self.0, id, &mut ptr))?;
        unsafe { Ok(CStr::from_ptr(ptr).to_str()?.to_owned()) }
    }

    pub fn output_name(&self, id: usize) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_model_output_name(self.0, id, &mut ptr))?;
        unsafe { Ok(CStr::from_ptr(ptr).to_str()?.to_owned()) }
    }

    pub fn set_output_names(&mut self, outputs: impl IntoIterator<Item = impl AsRef<str>>) -> Result<()> {
        let c_strings:Vec<CString> = outputs.into_iter().map(|a| Ok(CString::new(a.as_ref())?)).collect::<Result<_>>()?;
        let ptrs:Vec<_> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
        check!(sys::tract_model_set_output_names(self.0, c_strings.len(), ptrs.as_ptr()))?;
        Ok(())
    }

    pub fn input_fact(&self, id: usize) -> Result<Fact> {
        let mut ptr = null_mut();
        check!(sys::tract_model_input_fact(self.0, id, &mut ptr))?;
        Ok(Fact(ptr))
    }

    pub fn output_fact(&self, id: usize) -> Result<Fact> {
        let mut ptr = null_mut();
        check!(sys::tract_model_output_fact(self.0, id, &mut ptr))?;
        Ok(Fact(ptr))
    }

    pub fn declutter(&mut self) -> Result<()> {
        check!(sys::tract_model_declutter(self.0))?;
        Ok(())
    }

    pub fn optimize(&mut self) -> Result<()> {
        check!(sys::tract_model_optimize(self.0))?;
        Ok(())
    }

    pub fn into_decluttered(self) -> Result<Model> {
        check!(sys::tract_model_declutter(self.0))?;
        Ok(self)
    }

    pub fn into_optimized(self) -> Result<Model> {
        check!(sys::tract_model_optimize(self.0))?;
        Ok(self)
    }

    pub fn into_runnable(self) -> Result<Runnable> {
        let mut model = self;
        let mut runnable = null_mut();
        check!(sys::tract_model_into_runnable(&mut model.0, &mut runnable))?;
        let mut i = 0;
        let mut o = 0;
        check!(sys::tract_runnable_nbio(runnable, &mut i, &mut o))?;
        Ok(Runnable(runnable, i, o))
    }

    pub fn concretize_symbols(&mut self, values: impl IntoIterator<Item=(impl AsRef<str>, i64)>) -> Result<()> {
        let (names, values):(Vec<_>, Vec<_>) = values.into_iter().unzip();
        let c_strings:Vec<CString> = names.into_iter().map(|a| Ok(CString::new(a.as_ref())?)).collect::<Result<_>>()?;
        let ptrs:Vec<_> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
        check!(sys::tract_model_concretize_symbols(self.0, ptrs.len(), ptrs.as_ptr(), values.as_ptr()))?;
        Ok(())
    }

    pub fn pulse(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<()> {
        let name = CString::new(name.as_ref())?;
        let value = CString::new(value.as_ref())?;
        check!(sys::tract_model_pulse_simple(&mut self.0, name.as_ptr(), value.as_ptr()))?;
        Ok(())
    }

    pub fn cost_json(&self) -> Result<String> {
        let input:Option<Vec<Value>> = None;
        self.profile_json(input)
    }

    pub fn profile_json<I, V, E>(&self, inputs: Option<I>) -> Result<String>
        where I: IntoIterator<Item = V>,
              V: TryInto<Value, Error = E>,
              E: Into<anyhow::Error>
    {
        let inputs = if let Some(inputs) =  inputs {
            let inputs = inputs.into_iter().map(|i| i.try_into().map_err(|e| e.into())).collect::<Result<Vec<Value>>>()?;
            anyhow::ensure!(self.input_count()? == inputs.len());
            Some(inputs)
        } else { None };
        let mut iptrs:Option<Vec<*mut sys::TractValue>> = inputs.as_ref().map(|is| is.iter().map(|v| v.0).collect());
        let mut json : *mut i8 = null_mut();
        let values = iptrs.as_mut().map(|it| it.as_mut_ptr()).unwrap_or(null_mut());
        check!(sys::tract_model_profile_json(self.0, values, &mut json))?;
        anyhow::ensure!(!json.is_null());
        unsafe {
            let s = CStr::from_ptr(json).to_owned();
            sys::tract_free_cstring(json);
            Ok(s.to_str()?.to_owned())
        }
    }

    pub fn property_keys(&self) -> Result<Vec<String>> {
        let mut len = 0;
        check!(sys::tract_model_property_count(self.0, &mut len))?;
        let mut keys = vec!(null_mut(); len);
        check!(sys::tract_model_property_names(self.0, keys.as_mut_ptr()))?;
        unsafe {
            keys.into_iter().map(|pc| Ok(CStr::from_ptr(pc).to_str()?.to_owned())).collect()
        }
    }

    pub fn property(&self, name: impl AsRef<str>) -> Result<Value> {
        let mut v = null_mut();
        let name = CString::new(name.as_ref())?;
        check!(sys::tract_model_property(self.0, name.as_ptr(), &mut v))?;
        Ok(Value(v))
    }
}

// RUNNABLE
wrapper!(Runnable, TractRunnable, tract_runnable_release, usize, usize);

impl Runnable {
    pub fn run<I, V, E>(&self, inputs: I) -> Result<Vec<Value>> 
        where I: IntoIterator<Item = V>,
              V: TryInto<Value, Error = E>,
              E: Into<anyhow::Error>
    {
        self.spawn_state()?.run(inputs)
    }

    pub fn spawn_state(&self) -> Result<State> {
        let mut state = null_mut();
        check!(sys::tract_runnable_spawn_state(self.0, &mut state))?;
        Ok(State(state, self.1, self.2))
    }
}

// STATE
wrapper!(State, TractState, tract_state_destroy, usize, usize);

impl State {
    pub fn run<I, V, E>(&self, inputs: I) -> Result<Vec<Value>> 
        where I: IntoIterator<Item = V>,
              V: TryInto<Value, Error = E>,
              E: Into<anyhow::Error>
    {
        let inputs = inputs.into_iter().map(|i| i.try_into().map_err(|e| e.into())).collect::<Result<Vec<Value>>>()?;
        anyhow::ensure!(inputs.len() == self.1);
        let mut outputs = vec![null_mut(); self.2];
        let mut inputs: Vec<_> = inputs.iter().map(|v| v.0).collect();
        check!(sys::tract_state_run(self.0, inputs.as_mut_ptr(), outputs.as_mut_ptr()))?;
        let outputs = outputs.into_iter().map(|o| Value(o)).collect();
        Ok(outputs)
    }
}

// VALUE
wrapper!(Value, TractValue, tract_value_destroy);

impl Value {
    pub fn from_shape_and_slice<T: TractProxyDatumType>(shape: &[usize], data: &[T]) -> Result<Value> {
        anyhow::ensure!(data.len() == shape.iter().product());
        let mut value = null_mut();
        check!(sys::tract_value_create(
            T::c_repr(),
            shape.len(),
            shape.as_ptr(),
            data.as_ptr() as _,
            &mut value
        ))?;
        Ok(Value(value))
    }

    pub fn as_parts<'a, T: TractProxyDatumType>(&'a self) -> Result<(&'a [usize], &'a [T])> {
        let mut rank = 0;
        let mut dt = 0;
        let mut shape = null();
        let mut data = null();
        check!(sys::tract_value_inspect(self.0, &mut dt, &mut rank, &mut shape, &mut data))?;
        anyhow::ensure!(dt == T::c_repr());
        unsafe {
            let shape = std::slice::from_raw_parts(shape, rank);
            let len = shape.iter().product();
            let data = std::slice::from_raw_parts(data as *const T, len);
            Ok((shape, data))
        }
    }

    pub fn view<'a, T: TractProxyDatumType>(&self) -> Result<ndarray::ArrayViewD<T>> {
        let (shape, data) = self.as_parts()?;
        Ok(ndarray::ArrayViewD::from_shape(shape, data)?)
    }
}

impl<T,S,D> TryFrom<ndarray::ArrayBase<S, D>> for Value 
where T:TractProxyDatumType, S: RawData<Elem=T> + Data, D: Dimension
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

impl<'a, T: TractProxyDatumType> TryFrom<&'a Value> for ndarray::ArrayViewD<'a, T> {
    type Error = anyhow::Error;
    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        value.view()
    }
}

// FACT
wrapper!(Fact, TractFact, tract_fact_destroy);

impl Fact {
    pub fn new(model: &mut Model, spec: impl ToString) -> Result<Fact> {
        let cstr = CString::new(spec.to_string())?;
        let mut fact = null_mut();
        check!(sys::tract_fact_parse(model.0, cstr.as_ptr(), &mut fact))?;
        Ok(Fact(fact))
    }

    fn dump(&self) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_fact_dump(self.0, &mut ptr))?;
        unsafe { Ok(CStr::from_ptr(ptr).to_str()?.to_owned()) }
    }
}

impl Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dump().unwrap())
    }
}

// INFERENCE FACT
wrapper!(InferenceFact, TractInferenceFact, tract_inference_fact_destroy);

impl InferenceFact {
    pub fn new(model: &mut InferenceModel, spec: impl ToString) -> Result<InferenceFact> {
        let cstr = CString::new(spec.to_string())?;
        let mut fact = null_mut();
        check!(sys::tract_inference_fact_parse(model.0, cstr.as_ptr(), &mut fact))?;
        Ok(InferenceFact(fact))
    }

    fn dump(&self) -> Result<String> {
        let mut ptr = null_mut();
        check!(sys::tract_inference_fact_dump(self.0, &mut ptr))?;
        unsafe { Ok(CStr::from_ptr(ptr).to_str()?.to_owned()) }
    }
}

impl Display for InferenceFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dump().unwrap())
    }
}

pub trait AsFact<M, F> {
    fn as_fact(&self, model: &mut M) -> Result<Bow<F>>;
}

impl AsFact<InferenceModel, InferenceFact> for InferenceFact {
    fn as_fact(&self, _model: &mut InferenceModel) -> Result<Bow<InferenceFact>> {
        Ok(Bow::Borrowed(self))
    }
}

impl AsFact<InferenceModel, InferenceFact> for &str {
    fn as_fact(&self, model: &mut InferenceModel) -> Result<Bow<InferenceFact>> {
        Ok(Bow::Owned(InferenceFact::new(model, self)?))
    }
}

impl AsFact<InferenceModel, InferenceFact> for () {
    fn as_fact(&self, model: &mut InferenceModel) -> Result<Bow<InferenceFact>> {
        Ok(Bow::Owned(InferenceFact::new(model, "")?))
    }
}

impl AsFact<InferenceModel, InferenceFact> for Option<&str> {
    fn as_fact(&self, model: &mut InferenceModel) -> Result<Bow<InferenceFact>> {
        if let Some(it) = self {
            Ok(Bow::Owned(InferenceFact::new(model, it)?))
        } else {
            Ok(Bow::Owned(InferenceFact::new(model, "")?))
        }
    }
}

impl AsFact<Model, Fact> for Fact {
    fn as_fact(&self, _model: &mut Model) -> Result<Bow<Fact>> {
        Ok(Bow::Borrowed(self))
    }
}

impl<S: AsRef<str>> AsFact<Model, Fact> for S {
    fn as_fact(&self, model: &mut Model) -> Result<Bow<Fact>> {
        Ok(Bow::Owned(Fact::new(model, self.as_ref())?))
    }
}

pub trait TractProxyDatumType: Clone {
    fn c_repr() -> u32;
}

macro_rules! impl_datum_type {
    ($ty:ty, $c_repr:literal) => {
        impl TractProxyDatumType for $ty {
            fn c_repr() -> u32 { $c_repr}
        }
    }
}

impl_datum_type!(bool, 0x01);
impl_datum_type!(u8, 0x11);
impl_datum_type!(u16, 0x12);
impl_datum_type!(u32, 0x14);
impl_datum_type!(u64, 0x18);
impl_datum_type!(i8, 0x21);
impl_datum_type!(i16, 0x22);
impl_datum_type!(i32, 0x24);
impl_datum_type!(i64, 0x28);
impl_datum_type!(half::f16, 0x32);
impl_datum_type!(f32, 0x34);
impl_datum_type!(f64, 0x38);
