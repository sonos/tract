use std::ffi::{CStr, CString};
use std::fmt::Display;
use std::os::unix::prelude::OsStrExt;
use std::path::Path;
use std::ptr::{null, null_mut};
use std::str::FromStr;

use ndarray::Dimension;
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
}

// MODEL
wrapper!(Model, TractModel, tract_model_destroy);

impl Model {
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
}

// RUNNABLE
wrapper!(Runnable, TractRunnable, tract_runnable_release, usize, usize);

impl Runnable {
    pub fn run(&self, inputs: impl AsRef<[Value]>) -> Result<Vec<Value>> {
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
    pub fn run(&mut self, inputs: impl AsRef<[Value]>) -> Result<Vec<Value>> {
        let inputs = inputs.as_ref();
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
    pub fn from_shape_and_slice(shape: &[usize], data: &[f32]) -> Result<Value> {
        anyhow::ensure!(data.len() == shape.iter().product());
        let mut value = null_mut();
        check!(sys::tract_value_create(
            sys::TractDatumType_TRACT_DATUM_TYPE_F32,
            shape.len(),
            shape.as_ptr(),
            data.as_ptr() as _,
            &mut value
        ))?;
        Ok(Value(value))
    }

    pub fn as_parts<'a>(&'a self) -> Result<(&'a [usize], &'a [f32])> {
        let mut rank = 0;
        let mut dt = 0;
        let mut shape = null();
        let mut data = null();
        check!(sys::tract_value_inspect(self.0, &mut dt, &mut rank, &mut shape, &mut data))?;
        anyhow::ensure!(dt == sys::TractDatumType_TRACT_DATUM_TYPE_F32);
        unsafe {
            let shape = std::slice::from_raw_parts(shape, rank);
            let len = shape.iter().product();
            let data = std::slice::from_raw_parts(data as *const f32, len);
            Ok((shape, data))
        }
    }
}

impl<'a, D: Dimension> TryFrom<ndarray::ArrayView<'a, f32, D>> for Value {
    type Error = anyhow::Error;
    fn try_from(view: ndarray::ArrayView<'a, f32, D>) -> Result<Value> {
        if let Some(slice) = view.as_slice_memory_order() {
            Value::from_shape_and_slice(view.shape(), slice)
        } else {
            let slice: Vec<_> = view.iter().copied().collect();
            Value::from_shape_and_slice(view.shape(), &slice)
        }
    }
}

impl<'a> TryFrom<&'a Value> for ndarray::ArrayViewD<'a, f32> {
    type Error = anyhow::Error;
    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        let (shape, data) = value.as_parts()?;
        Ok(ndarray::ArrayViewD::from_shape(shape, data)?)
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
