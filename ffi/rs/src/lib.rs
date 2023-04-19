use std::ffi::{CStr, CString};
use std::os::unix::prelude::OsStrExt;
use std::path::Path;

use ndarray::Dimension;
use tract_rs_sys as sys;

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

pub fn nnef() -> anyhow::Result<Nnef> {
    Nnef::new()
}

impl Nnef {
    pub fn new() -> anyhow::Result<Nnef> {
        let mut nnef = std::ptr::null_mut();
        check!(sys::tract_nnef_create(&mut nnef))?;
        Ok(Nnef(nnef))
    }

    pub fn model_for_path(&self, path: impl AsRef<Path>) -> anyhow::Result<Model> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes().to_vec())?;
        let mut model = std::ptr::null_mut();
        check!(sys::tract_nnef_model_for_path(self.0, path.as_ptr(), &mut model))?;
        Ok(Model(model))
    }
}

// MODEL
wrapper!(Model, TractModel, tract_model_destroy);

impl Model {
    pub fn into_optimized(self) -> anyhow::Result<Model> {
        check!(sys::tract_model_optimize(self.0))?;
        Ok(self)
    }

    pub fn into_runnable(self) -> anyhow::Result<Runnable> {
        let mut model = self;
        let mut runnable = std::ptr::null_mut();
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
    pub fn run(&self, inputs: impl AsRef<[Value]>) -> anyhow::Result<Vec<Value>> {
        self.spawn_state()?.run(inputs)
    }

    pub fn spawn_state(&self) -> anyhow::Result<State> {
        let mut state = std::ptr::null_mut();
        check!(sys::tract_runnable_spawn_state(self.0, &mut state))?;
        Ok(State(state, self.1, self.2))
    }
}

// STATE
wrapper!(State, TractState, tract_state_destroy, usize, usize);

impl State {
    pub fn run(&mut self, inputs: impl AsRef<[Value]>) -> anyhow::Result<Vec<Value>> {
        let inputs = inputs.as_ref();
        anyhow::ensure!(inputs.len() == self.1);
        let mut outputs = vec!(std::ptr::null_mut(); self.2);
        let mut inputs:Vec<_> = inputs.iter().map(|v| v.0).collect();
        check!(sys::tract_state_run(self.0, inputs.as_mut_ptr(), outputs.as_mut_ptr()))?;
        let outputs = outputs.into_iter().map(|o| Value(o)).collect();
        Ok(outputs)
    }
}

// VALUE
wrapper!(Value, TractValue, tract_value_destroy);

impl Value {
    pub fn from_shape_and_slice(shape: &[usize], data: &[f32]) -> anyhow::Result<Value> {
        anyhow::ensure!(data.len() == shape.iter().product());
        let mut value = std::ptr::null_mut();
        check!(sys::tract_value_create(
            sys::TractDatumType_TRACT_DATUM_TYPE_F32,
            shape.len(),
            shape.as_ptr(),
            data.as_ptr() as _,
            &mut value
        ))?;
        Ok(Value(value))
    }

    pub fn as_parts<'a>(&'a self) -> anyhow::Result<(&'a [usize], &'a [f32])> {
        let mut rank = 0;
        let mut dt = 0;
        let mut shape = std::ptr::null();
        let mut data = std::ptr::null();
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
    fn try_from(view: ndarray::ArrayView<'a, f32, D>) -> anyhow::Result<Value> {
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
