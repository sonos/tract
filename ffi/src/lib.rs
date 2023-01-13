#![allow(clippy::missing_safety_doc)]

use anyhow::Context;
use std::cell::RefCell;
use std::convert::TryFrom;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

use tract_nnef::internal as native;
use tract_nnef::tract_core::prelude::*;

use tract_onnx::prelude::{self as onnx, InferenceFact};
use tract_onnx::prelude::InferenceModelExt;

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

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum TractDatumType {
    TRACT_DATUM_TYPE_BOOL = 0x01,
    TRACT_DATUM_TYPE_U8 = 0x11,
    TRACT_DATUM_TYPE_U16 = 0x12,
    TRACT_DATUM_TYPE_U32 = 0x14,
    TRACT_DATUM_TYPE_U64 = 0x18,
    TRACT_DATUM_TYPE_I8 = 0x21,
    TRACT_DATUM_TYPE_I16 = 0x22,
    TRACT_DATUM_TYPE_I32 = 0x24,
    TRACT_DATUM_TYPE_I64 = 0x28,
    TRACT_DATUM_TYPE_F16 = 0x32,
    TRACT_DATUM_TYPE_F32 = 0x34,
    TRACT_DATUM_TYPE_F64 = 0x38,
    TRACT_DATUM_TYPE_COMPLEX_I16 = 0x42,
    TRACT_DATUM_TYPE_COMPLEX_I32 = 0x44,
    TRACT_DATUM_TYPE_COMPLEX_I64 = 0x48,
    TRACT_DATUM_TYPE_COMPLEX_F16 = 0x52,
    TRACT_DATUM_TYPE_COMPLEX_F32 = 0x54,
    TRACT_DATUM_TYPE_COMPLEX_F64 = 0x58,
}

impl From<TractDatumType> for DatumType {
    fn from(it: TractDatumType) -> Self {
        use DatumType::*;
        use TractDatumType::*;
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
            TRACT_DATUM_TYPE_COMPLEX_I16 => ComplexI16,
            TRACT_DATUM_TYPE_COMPLEX_I32 => ComplexI32,
            TRACT_DATUM_TYPE_COMPLEX_I64 => ComplexI64,
            TRACT_DATUM_TYPE_COMPLEX_F16 => ComplexF16,
            TRACT_DATUM_TYPE_COMPLEX_F32 => ComplexF32,
            TRACT_DATUM_TYPE_COMPLEX_F64 => ComplexF64,
        }
    }
}

impl TryFrom<DatumType> for TractDatumType {
    type Error = TractError;
    fn try_from(it: DatumType) -> TractResult<Self> {
        use DatumType::*;
        use TractDatumType::*;
        match it {
            Bool => Ok(TRACT_DATUM_TYPE_BOOL),
            U8 => Ok(TRACT_DATUM_TYPE_U8),
            U16 => Ok(TRACT_DATUM_TYPE_U16),
            U32 => Ok(TRACT_DATUM_TYPE_U32),
            U64 => Ok(TRACT_DATUM_TYPE_U64),
            I8 => Ok(TRACT_DATUM_TYPE_I8),
            I16 => Ok(TRACT_DATUM_TYPE_I16),
            I32 => Ok(TRACT_DATUM_TYPE_I32),
            I64 => Ok(TRACT_DATUM_TYPE_I64),
            F16 => Ok(TRACT_DATUM_TYPE_F16),
            F32 => Ok(TRACT_DATUM_TYPE_F32),
            F64 => Ok(TRACT_DATUM_TYPE_F64),
            ComplexI16 => Ok(TRACT_DATUM_TYPE_COMPLEX_I16),
            ComplexI32 => Ok(TRACT_DATUM_TYPE_COMPLEX_I32),
            ComplexI64 => Ok(TRACT_DATUM_TYPE_COMPLEX_I64),
            ComplexF16 => Ok(TRACT_DATUM_TYPE_COMPLEX_F16),
            ComplexF32 => Ok(TRACT_DATUM_TYPE_COMPLEX_F32),
            ComplexF64 => Ok(TRACT_DATUM_TYPE_COMPLEX_F64),
            _ => anyhow::bail!("tract C bindings do not support {:?} type", it),
        }
    }
}

thread_local! {
    pub(crate) static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
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

/// Retrieve the last error that happened in this thread. A function encountered an error if
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

#[no_mangle]
pub extern "C" fn tract_version() -> *const std::ffi::c_char {
    unsafe {
        CStr::from_bytes_with_nul_unchecked(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes())
            .as_ptr()
    }
}

// NNEF
pub struct TractNnef(native::Nnef);

/// Creates an instance of an NNEF framework and parser that can be used to load models.
///
/// The returned object should be destroyed with `tract_nnef_destroy` once the model
/// has been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_create(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        *nnef = Box::into_raw(Box::new(TractNnef(tract_nnef::nnef())));
        Ok(())
    })
}

/// Destroy the NNEF parser. It is safe to detroy the NNEF parser once the model had been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_destroy(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        if nnef.is_null() || (*nnef).is_null() {
            anyhow::bail!("Trying to destroy a null Nnef object");
        }
        let _ = Box::from_raw(*nnef);
        *nnef = std::ptr::null_mut();
        Ok(())
    })
}

/// Parse and load an NNEF model as a tract TypedModel.
///
/// `path` is a null-terminated utf-8 string pointer. It can be an archive (tar or tar.gz file) or a
/// directory.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_model_for_path(
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

// ONNX
pub struct TractOnnx(tract_onnx::Onnx);

/// Creates an instance of an ONNX framework and parser that can be used to load models.
///
/// The returned object should be destroyed with `tract_nnef_destroy` once the model
/// has been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_create(ptr: *mut *mut TractOnnx) -> TRACT_RESULT {
    wrap(|| unsafe {
        *ptr = Box::into_raw(Box::new(TractOnnx(onnx::onnx())));
        Ok(())
    })
}

/// Destroy the NNEF parser. It is safe to detroy the NNEF parser once the model had been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_destroy(ptr: *mut *mut TractOnnx) -> TRACT_RESULT {
    wrap(|| unsafe {
        if ptr.is_null() || (*ptr).is_null() {
            anyhow::bail!("Trying to destroy a null Onnx object");
        }
        let _ = Box::from_raw(*ptr);
        *ptr = std::ptr::null_mut();
        Ok(())
    })
}

/// Parse and load an ONNX model as a tract InferenceModel.
///
/// `path` is a null-terminated utf-8 string pointer. It must point to a `.onnx` model file.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_model_for_path(
    onnx: &TractOnnx,
    path: *const c_char,
    model: *mut *mut TractInferenceModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        *model = std::ptr::null_mut();
        let path = CStr::from_ptr(path).to_str()?;
        let m = Box::new(TractInferenceModel(
            onnx.0.model_for_path(path).with_context(|| format!("opening file {:?}", path))?,
        ));
        *model = Box::into_raw(m);
        Ok(())
    })
}

// INFERENCE MODEL
pub struct TractInferenceModel(onnx::InferenceModel);

#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_set_input_fact(
    model: *mut TractInferenceModel,
    input_id: usize,
    fact: *const TractInferenceFact
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() {
            anyhow::bail!("Trying to alter a null inference model")
        }
        let f = fact.as_ref().map(|f| &f.0).cloned().unwrap_or_default();
        (*model).0.set_input_fact(input_id, f)?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_set_output_fact(
    model: *mut TractInferenceModel,
    output_id: usize,
    fact: *const TractInferenceFact
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() {
            anyhow::bail!("Trying to alter a null inference model")
        }
        let f = fact.as_ref().map(|f| &f.0).cloned().unwrap_or_default();
        (*model).0.set_output_fact(output_id, f)?;
        Ok(())
    })
}

/// Convenience function to obtain an optimized TypedModel from an InferenceModel.
///
/// This function takes ownership of the InferenceModel `model` whether it succeeds
/// or not. `tract_inference_model_destroy` must not be used on `model`.
///
/// On the other hand, caller will be owning the newly created `optimized` model.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_into_optimized(
    model: *mut *mut TractInferenceModel,
    optimized: *mut *mut TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() {
            anyhow::bail!("Trying to convert null inference model")
        }
        let m = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        let model = m.0.into_optimized()?;
        *optimized = Box::into_raw(Box::new(TractModel(model))) as _;
        Ok(())
    })
}

/// Destroy an InferenceModel.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_destroy(
    model: *mut *mut TractInferenceModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() || (*model).is_null() {
            anyhow::bail!("Trying to destroy a null InferenceModel");
        }
        let _ = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        Ok(())
    })
}
// TYPED MODEL

pub struct TractModel(TypedModel);

/// Optimize a TypedModel in-place.
#[no_mangle]
pub unsafe extern "C" fn tract_model_optimize(model: *mut TractModel) -> TRACT_RESULT {
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
/// This function transfers ownership of the `model` argument to the newly-created `runnable` model.
///
/// Runnable are reference counted. When done, it should be released with `tract_runnable_release`.
#[no_mangle]
pub unsafe extern "C" fn tract_model_into_runnable(
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

/// Destroy a TypedModel.
#[no_mangle]
pub unsafe extern "C" fn tract_model_destroy(model: *mut *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() || (*model).is_null() {
            anyhow::bail!("Trying to destroy a null Model");
        }
        let _ = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        Ok(())
    })
}

// RUNNABLE MODEL
pub struct TractRunnable(Arc<native::TypedRunnableModel<native::TypedModel>>);

/// Spawn a session state from a runnable model.
///
/// This function does not take ownership of the `runnable` object, it can be used again to spawn
/// other state instances. The runnable object is internally reference counted, it will be
/// kept alive as long as any associated `State` exists (or as long as the `runnable` is not
/// explicitely release with `tract_runnable_release`).
///
/// `state` is a newly-created object. It should ultimately be detroyed with `tract_state_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_spawn_state(
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

/// Convenience function to run a stateless model.
///
/// `inputs` is a pointer to an pre-existing array of input TractValue. Its length *must* be equal
/// to the number of inputs of the models. The function does not take ownership of the input
/// values.
/// `outputs` is a pointer to a pre-existing array of TractValue pointers that will be overwritten
/// with pointers to outputs values. These values are under the responsiblity of the caller, it
/// will have to release them with `tract_value_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_run(
    runnable: *mut TractRunnable,
    inputs: *mut *mut TractValue,
    outputs: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if runnable.is_null() {
            anyhow::bail!("Trying to convert null model")
        }
        let runnable = runnable.as_ref().unwrap();
        let mut s = native::TypedSimpleState::new(runnable.0.clone())?;
        state_run(&mut s, inputs, outputs)
    })
}

/// Query a runnable model input and output counts.
///
/// It can be called with null as `inputs` or `outputs` if only one count is required.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_nbio(
    runnable: *mut TractRunnable,
    inputs: *mut usize,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if runnable.is_null() {
            anyhow::bail!("Trying to convert null model")
        }
        let runnable = runnable.as_ref().unwrap();
        if !inputs.is_null() {
            *inputs = runnable.0.model().inputs.len();
        }
        if !outputs.is_null() {
            *outputs = runnable.0.model().outputs.len();
        }
        Ok(())
    })
}

unsafe fn state_run(
    state: &mut NativeState,
    inputs: *mut *mut TractValue,
    outputs: *mut *mut TractValue,
) -> TractResult<()> {
    if inputs.is_null() {
        anyhow::bail!("Null pointer input")
    }
    if outputs.is_null() {
        anyhow::bail!("Null pointer output")
    }
    let input_len = state.model().inputs.len();
    let values =
        std::slice::from_raw_parts(inputs, input_len).iter().map(|tv| (**tv).0.clone()).collect();
    let values = state.run(values)?;
    for (i, value) in values.into_iter().enumerate() {
        *(outputs.add(i)) = Box::into_raw(Box::new(TractValue(value)))
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn tract_runnable_release(runnable: *mut *mut TractRunnable) -> TRACT_RESULT {
    wrap(|| unsafe {
        if runnable.is_null() || (*runnable).is_null() {
            anyhow::bail!("Trying to destroy a null Runnable");
        }
        let _ = Box::from_raw(*runnable);
        *runnable = std::ptr::null_mut();
        Ok(())
    })
}

// VALUE
pub struct TractValue(TValue);

/// Create a TractValue (aka tensor) from caller data and metadata.
///
/// This call copies the data into tract space. All the pointers only need to be alive for the
/// duration of the call.
///
/// rank is the number of dimensions of the :ne
#[no_mangle]
pub unsafe extern "C" fn tract_value_create(
    datum_type: TractDatumType,
    rank: usize,
    shape: *const usize,
    data: *mut c_void,
    value: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        let dt: DatumType = datum_type.into();
        let shape = std::slice::from_raw_parts(shape, rank);
        let len = shape.iter().product::<usize>();
        let content = std::slice::from_raw_parts(data as *const u8, len * dt.size_of());
        let it = Tensor::from_raw_dt(dt, shape, content)?;
        *value = Box::into_raw(Box::new(TractValue(it.into_tvalue())));
        Ok(())
    })
}

/// Destroy a value.
#[no_mangle]
pub unsafe extern "C" fn tract_value_destroy(value: *mut *mut TractValue) -> TRACT_RESULT {
    wrap(|| unsafe {
        if value.is_null() || (*value).is_null() {
            anyhow::bail!("Trying to destroy a null Value");
        }
        let _ = Box::from_raw(*value);
        *value = std::ptr::null_mut();
        Ok(())
    })
}

/// Inspect part of a value. Except `value`, all argument pointers can be null if only some specific bits
/// are required.
#[no_mangle]
pub unsafe extern "C" fn tract_value_inspect(
    value: *mut TractValue,
    datum_type: *mut TractDatumType,
    rank: *mut usize,
    shape: *mut *const usize,
    data: *mut *const c_void,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if value.is_null() {
            anyhow::bail!("Trying to inspect a null Value");
        }
        let value: &TValue = &(*value).0;
        if !datum_type.is_null() {
            *datum_type = value.datum_type().try_into()?;
        }
        if !rank.is_null() {
            *rank = value.rank();
        }
        if !shape.is_null() {
            *shape = value.shape().as_ptr();
        }
        if !data.is_null() {
            *data = value.as_ptr_unchecked::<u8>() as _;
        }
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

/// Convenience function running a pass of a statefull model.
///
/// `inputs` is a pointer to an pre-existing array of input TractValue. Its length *must* be equal
/// to the number of inputs of the models. The function does not take ownership of the input
/// values.
/// `outputs` is a pointer to a pre-existing array of TractValue pointers that will be overwritten
/// with pointers to outputs values. These values are under the responsiblity of the caller, it
/// will have to release them with `tract_value_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_state_run(
    state: *mut TractState,
    inputs: *mut *mut TractValue,
    outputs: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to run a null State");
        }
        let s = state.as_mut().unwrap();
        state_run(&mut s.0, inputs, outputs)
    })
}

/*
#[no_mangle]
pub unsafe extern "C" fn tract_state_set_input(
    state: *mut TractState,
    input_id: usize,
    value: *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to set input on a null State");
        }
        if value.is_null() {
            anyhow::bail!("Trying to set input to a null value");
        }
        let state = state.as_mut().unwrap();
        let value = value.as_ref().unwrap();
        state.0.set_input(input_id, value.0.clone())?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_state_exec(state: *mut TractState) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to exec a null State");
        }
        let state = state.as_mut().unwrap();
        state.0.exec()?;
        Ok(())
    })
}

/// Get an output tensor from the state.
#[no_mangle]
pub unsafe extern "C" fn tract_state_output(
    state: *mut TractState,
    output_id: usize,
    tensor: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to exec a null State");
        }
        let state = state.as_mut().unwrap();
        let value = state.0.output(output_id)?;
        *tensor = Box::into_raw(Box::new(TractValue(value.clone())));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_state_reset_turn(state: *mut TractState) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Trying to reset turn on a null State");
        }
        let state = state.as_mut().unwrap();
        state.0.reset_turn()?;
        Ok(())
    })
}
*/

#[no_mangle]
pub unsafe extern "C" fn tract_state_destroy(state: *mut *mut TractState) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() || (*state).is_null() {
            anyhow::bail!("Trying to destroy a null State");
        }
        let _ = Box::from_raw(*state);
        *state = std::ptr::null_mut();
        Ok(())
    })
}

// INFERENCE FACT

pub struct TractInferenceFact(InferenceFact);

#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_parse(
    model: *mut TractInferenceModel,
    spec: *const c_char,
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() {
            anyhow::bail!("Trying to build an inference fact for a null model");
        }
        let spec = CStr::from_ptr(spec).to_str()?;
        let model = model.as_ref().unwrap();
        let f = tract_libcli::tensor::parse_spec(&model.0.symbol_table, spec)?;
        *fact = Box::into_raw(Box::new(TractInferenceFact(f)));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_destroy(fact: *mut *mut TractInferenceFact) -> TRACT_RESULT {
    wrap(|| unsafe {
        if fact.is_null() || (*fact).is_null() {
            anyhow::bail!("Trying to destroy a null InferenceFact");
        }
        let _ = Box::from_raw(*fact);
        *fact = std::ptr::null_mut();
        Ok(())
    })
}


// MISC

// HELPERS
