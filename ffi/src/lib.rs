use anyhow::Context;
use dlpackrs::ffi::DLTensor;
use dlpackrs::{DataType, Device};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;
use tract_nnef::tract_ndarray::RawArrayView;

use tract_nnef::internal as native;
use tract_nnef::tract_core::prelude::*;

/// Used as a return type of functions that can encounter errors.
/// If the function encountered an error, you can retrieve it using the `tract_get_last_error`
/// function
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Eq)]
pub enum TRACT_RESULT {
    /// The function returned successfully
    TRACT_RESULT_OK = 0,
    /// The function returned an error
    TRACT_RESULT_KO = 1,
}

thread_local! {
    pub(crate) static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
    pub(crate) static SHAPE_ARRAY: RefCell<TVec<i64>> = RefCell::new(tvec!());
    pub(crate) static OUTPUTS: RefCell<TVec<TValue>> = RefCell::new(tvec!());
}

fn wrap<F: FnOnce() -> anyhow::Result<()>>(func: F) -> TRACT_RESULT {
    match func() {
        Ok(_) => TRACT_RESULT::TRACT_RESULT_OK,
        Err(e) => {
            let msg = format!("{:?}", e);
            if std::env::var("TRACT_ERROR_STDERR").is_ok() {
                eprintln!("{}", msg);
            }
            LAST_ERROR.with(|p| {
                *p.borrow_mut() = Some(CString::new(msg).unwrap_or_else(|_| {
                    CString::new("tract error message contains 0, can't convert to CString")
                        .unwrap()
                }))
            });
            TRACT_RESULT::TRACT_RESULT_KO
        }
    }
}

/// Used to retrieve the last error that happened in this thread. A function encountered an error if
/// its return type is of type `TRACT_RESULT` and it returned `TRACT_RESULT_KO`.
///
/// # Return value
///  It returns a pointer to a null-terminated UTF-8 string that will contain the error description.
///  Rust side keeps ownership of the buffer. It will be valid as long as no other tract calls is
///  performed by the thread.
///  If no error occured, null is returned.
#[no_mangle]
pub extern "C" fn tract_get_last_error() -> *const std::ffi::c_char {
    LAST_ERROR.with(|msg| msg.borrow().as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()))
}

// NNEF

pub struct TractNnef(native::Nnef);

#[no_mangle]
pub extern "C" fn tract_nnef_create(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        *nnef = Box::into_raw(Box::new(TractNnef(tract_nnef::nnef())));
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_nnef_destroy(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        if nnef.is_null() || (*nnef).is_null() {
            anyhow::bail!("Trying to destroy a null Nnef object");
        }
        let _ = Box::from_raw(*nnef);
        *nnef = std::ptr::null_mut();
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_nnef_model_for_path(
    nnef: &TractNnef,
    path: *const c_char,
    model: *mut *mut TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        *model = std::ptr::null_mut();
        let path = CStr::from_ptr(path).to_str()?;
        let m = Box::new(TractModel(
            nnef.0.model_for_path(path).with_context(|| format!("opening file {:?}", path))?,
        ));
        *model = Box::into_raw(m);
        Ok(())
    })
}

// TYPED MODEL

pub struct TractModel(TypedModel);

#[no_mangle]
pub extern "C" fn tract_model_optimize(model: *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        if let Some(model) = model.as_mut() {
            model.0.optimize()
        } else {
            anyhow::bail!("Trying to optimise null model")
        }
    })
}

/// Convert a TypedModel into a TypedRunnableModel.
///
/// This function transfers ownership of the model argument to the runnable model.
#[no_mangle]
pub extern "C" fn tract_model_into_runnable(
    model: *mut *mut TractModel,
    runnable: *mut *mut TractRunnable,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() {
            anyhow::bail!("Trying to convert null model")
        }
        let m = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        let runnable_model = m.0.into_runnable()?;
        *runnable = Box::into_raw(Box::new(TractRunnable(Arc::new(runnable_model)))) as _;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_model_destroy(model: *mut *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() || (*model).is_null() {
            anyhow::bail!("Trying to destroy a null Model");
        }
        let _ = Box::from_raw(*model);
        Ok(())
    })
}

// RUNNABLE MODEL
pub struct TractRunnable(Arc<native::TypedRunnableModel<native::TypedModel>>);

#[no_mangle]
pub extern "C" fn tract_runnable_spawn_state(
    runnable: *mut TractRunnable,
    state: *mut *mut TractState,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Null pointer for expected state return")
        }
        *state = std::ptr::null_mut();
        if runnable.is_null() {
            anyhow::bail!("Trying to convert null model")
        }
        let runnable = runnable.as_ref().unwrap();
        let s = native::TypedSimpleState::new(runnable.0.clone())?;
        *state = Box::into_raw(Box::new(TractState(s)));
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_runnable_run(
    runnable: *mut TractRunnable,
    input_len: usize,
    inputs: *mut DLTensor,
    output_len: usize,
    outputs: *mut DLTensor,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if runnable.is_null() {
            anyhow::bail!("Trying to convert null model")
        }
        if inputs.is_null() {
            anyhow::bail!("Null pointer input")
        }
        /*
        if outputs.is_null() {
            anyhow::bail!("Null pointer output")
        }
        */
        let runnable = runnable.as_ref().unwrap();
        let values = (0..input_len)
            .map(|ix| Ok(copy_tensor_to_tract(inputs.add(ix))?.into_tvalue()))
            .collect::<TractResult<TVec<TValue>>>()?;
        dbg!(values);
        anyhow::bail!("foo");
        let values = runnable.0.run(values)?;
        if values.len() != output_len {
            anyhow::bail!(
                "Wrong output number. output_len says {}, model returned {} outputs",
                output_len,
                values.len()
            )
        }
        OUTPUTS.with(|store| {
            let mut store = store.borrow_mut();
            *store = values;
            observe_tensors(&store, outputs)
        })
    })
}

#[no_mangle]
pub extern "C" fn tract_runnable_release(runnable: *mut *mut TractRunnable) -> TRACT_RESULT {
    wrap(|| unsafe {
        if runnable.is_null() || (*runnable).is_null() {
            anyhow::bail!("Trying to destroy a null Runnable");
        }
        let _ = Box::from_raw(*runnable);
        *runnable = std::ptr::null_mut();
        Ok(())
    })
}

// STATE
/// cbindgen:ignore
type NativeState = native::TypedSimpleState<
    native::TypedModel,
    Arc<native::TypedRunnableModel<native::TypedModel>>,
>;
pub struct TractState(NativeState);

#[no_mangle]
pub extern "C" fn tract_state_set_input(
    state: *mut TractState,
    input_id: usize,
    tensor: *mut DLTensor,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to set input on a null State");
        }
        let state = state.as_mut().unwrap();
        let tensor = copy_tensor_to_tract(tensor)?;
        state.0.set_input(input_id, tensor.into_tvalue())?;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_state_exec(state: *mut TractState) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to exec a null State");
        }
        let state = state.as_mut().unwrap();
        state.0.exec()?;
        Ok(())
    })
}

/// Borrow an output tensor from the state.
///
/// The borrowed data may become invalid when any function on tract API is called in the thread.
#[no_mangle]
pub extern "C" fn tract_state_output(
    state: *mut TractState,
    output_id: usize,
    tensor: *mut DLTensor,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to exec a null State");
        }
        let state = state.as_mut().unwrap();
        let value = state.0.output(output_id)?;
        observe_tensors(&[value.to_owned()], tensor)?;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_state_reset_turn(state: *mut TractState) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to reset turn on a null State");
        }
        let state = state.as_mut().unwrap();
        state.0.reset_turn()?;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_state_destroy(state: *mut *mut TractState) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() || (*state).is_null() {
            anyhow::bail!("Trying to destroy a null State");
        }
        let _ = Box::from_raw(*state);
        *state = std::ptr::null_mut();
        Ok(())
    })
}

// MISC

/// This ffi binding uses a couple of thread local variables to adapt tensors and DLTensor
/// semantics, specifically in the case of `tract_state_output` or `tract_runnable_run`. These
/// variables will stay in memory until one of the functions is called again, and the return
/// DLTensors will stay valid accordingly.
///
/// If memory is a concern and the DLTensor no longer needed, `tract_clear_dltensor_helpers`
/// can be called to free the thread local storage.
#[no_mangle]
pub extern "C" fn tract_clear_dltensor_storage() {
    SHAPE_ARRAY.with(|s| s.replace(tvec!()));
    OUTPUTS.with(|s| s.replace(tvec!()));
}


// HELPERS

/// cbindgen:ignore
fn copy_tensor_to_tract(tensor: *mut dlpackrs::ffi::DLTensor) -> TractResult<Tensor> {
    unsafe {
        let dlt = dlpackrs::Tensor::from_raw(tensor);
        assert!(dlt.dtype() == DataType::f32());
        assert!(dlt.strides().is_none());
        let arr = RawArrayView::from_shape_ptr(dlt.shape().unwrap(), dlt.data() as *mut f32);
        Ok(arr.deref_into_view().into_owned().into_tensor())
    }
}

/// dbinget ignore
unsafe fn observe_tensors(values: &[TValue], ffi: *mut DLTensor) -> TractResult<()> {
    SHAPE_ARRAY.with(|shape| {
        let mut shape = shape.borrow_mut();
        shape.clear();
        for (ix, value) in values.iter().enumerate() {
            let dlt = dlpackrs::Tensor::new(
                value.as_ptr::<f32>()? as _,
                Device::cpu(0),
                value.rank() as i32,
                DataType::f32(),
                shape.as_mut_ptr().add(shape.len()),
                std::ptr::null_mut(),
                0,
            );
            shape.extend(value.shape().iter().map(|i| *i as i64));
            *(ffi.add(ix)) = dlt.into_inner();
        }
        Ok(())
    })
}
