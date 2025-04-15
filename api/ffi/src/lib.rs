#![allow(clippy::missing_safety_doc)]

use anyhow::{Context, Result};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use tract_api::{
    AsFact, DatumType, InferenceModelInterface, ModelInterface, NnefInterface, OnnxInterface,
    RunnableInterface, StateInterface, ValueInterface,
};
use tract_rs::{State, Value};

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
    pub(crate) static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn wrap<F: FnOnce() -> anyhow::Result<()>>(func: F) -> TRACT_RESULT {
    match func() {
        Ok(_) => TRACT_RESULT::TRACT_RESULT_OK,
        Err(e) => {
            let msg = format!("{e:?}");
            if std::env::var("TRACT_ERROR_STDERR").is_ok() {
                eprintln!("{msg}");
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

/// Returns a pointer to a static buffer containing a null-terminated version string.
///
/// The returned pointer must not be freed.
#[no_mangle]
pub extern "C" fn tract_version() -> *const std::ffi::c_char {
    unsafe {
        CStr::from_bytes_with_nul_unchecked(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes())
            .as_ptr()
    }
}

/// Frees a string allocated by libtract.
#[no_mangle]
pub unsafe extern "C" fn tract_free_cstring(ptr: *mut std::ffi::c_char) {
    unsafe {
        if !ptr.is_null() {
            let _ = CString::from_raw(ptr);
        }
    }
}

macro_rules! check_not_null {
    ($($ptr:expr),*) => {
        $(
            if $ptr.is_null() {
                anyhow::bail!(concat!("Unexpected null pointer ", stringify!($ptr)));
            }
         )*
    }
}

macro_rules! release {
    ($ptr:expr) => {
        wrap(|| unsafe {
            check_not_null!($ptr, *$ptr);
            let _ = Box::from_raw(*$ptr);
            *$ptr = std::ptr::null_mut();
            Ok(())
        })
    };
}

// NNEF
pub struct TractNnef(tract_rs::Nnef);

/// Creates an instance of an NNEF framework and parser that can be used to load and dump NNEF models.
///
/// The returned object should be destroyed with `tract_nnef_destroy` once the model
/// has been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_create(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        *nnef = Box::into_raw(Box::new(TractNnef(tract_rs::nnef()?)));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_transform_model(
    nnef: *const TractNnef,
    model: * mut TractModel,
    transform_spec: *const i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, transform_spec);
        let transform_spec = CStr::from_ptr(transform_spec as _).to_str()?;
        (*nnef).0.transform_model(&mut (*model).0, transform_spec).with_context(|| format!("performing transform {transform_spec:?}"))?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_tract_core(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_tract_core()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_tract_extra(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_tract_extra()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_tract_transformers(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_tract_transformers()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_onnx(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_onnx()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_pulse(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_pulse()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_extended_identifier_syntax(
    nnef: *mut TractNnef,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_extended_identifier_syntax()
    })
}

/// Destroy the NNEF parser. It is safe to detroy the NNEF parser once the model had been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_destroy(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    release!(nnef)
}

/// Parse and load an NNEF model as a tract TypedModel.
///
/// `path` is a null-terminated utf-8 string pointer. It can be an archive (tar or tar.gz file) or a
/// directory.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_model_for_path(
    nnef: *const TractNnef,
    path: *const c_char,
    model: *mut *mut TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, path);
        *model = std::ptr::null_mut();
        let path = CStr::from_ptr(path).to_str()?;
        let m = Box::new(TractModel(
            (*nnef).0.model_for_path(path).with_context(|| format!("opening file {path:?}"))?,
        ));
        *model = Box::into_raw(m);
        Ok(())
    })
}

/// Dump a TypedModel as a NNEF tar file.
///
/// `path` is a null-terminated utf-8 string pointer to the `.tar` file to be created.
///
/// This function creates a plain, non-compressed, archive.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_write_model_to_tar(
    nnef: *const TractNnef,
    path: *const c_char,
    model: *const TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, path);
        let path = CStr::from_ptr(path).to_str()?;
        (*nnef).0.write_model_to_tar(path, &(*model).0)?;
        Ok(())
    })
}

/// Dump a TypedModel as a NNEF .tar.gz file.
///
/// `path` is a null-terminated utf-8 string pointer to the `.tar.gz` file to be created.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_write_model_to_tar_gz(
    nnef: *const TractNnef,
    path: *const c_char,
    model: *const TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, path);
        let path = CStr::from_ptr(path).to_str()?;
        (*nnef).0.write_model_to_tar_gz(path, &(*model).0)?;
        Ok(())
    })
}

/// Dump a TypedModel as a NNEF directory.
///
/// `path` is a null-terminated utf-8 string pointer to the directory to be created.
///
/// This function creates a plain, non-compressed, archive.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_write_model_to_dir(
    nnef: *const TractNnef,
    path: *const c_char,
    model: *const TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, path);
        let path = CStr::from_ptr(path).to_str()?;
        (*nnef).0.write_model_to_dir(path, &(*model).0)?;
        Ok(())
    })
}

// ONNX
pub struct TractOnnx(tract_rs::Onnx);

/// Creates an instance of an ONNX framework and parser that can be used to load models.
///
/// The returned object should be destroyed with `tract_nnef_destroy` once the model
/// has been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_create(onnx: *mut *mut TractOnnx) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(onnx);
        *onnx = Box::into_raw(Box::new(TractOnnx(tract_rs::onnx()?)));
        Ok(())
    })
}

/// Destroy the NNEF parser. It is safe to detroy the NNEF parser once the model had been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_destroy(onnx: *mut *mut TractOnnx) -> TRACT_RESULT {
    release!(onnx)
}

/// Parse and load an ONNX model as a tract InferenceModel.
///
/// `path` is a null-terminated utf-8 string pointer. It must point to a `.onnx` model file.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_model_for_path(
    onnx: *const TractOnnx,
    path: *const c_char,
    model: *mut *mut TractInferenceModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(onnx, path, model);
        *model = std::ptr::null_mut();
        let path = CStr::from_ptr(path).to_str()?;
        let m = Box::new(TractInferenceModel((*onnx).0.model_for_path(path)?));
        *model = Box::into_raw(m);
        Ok(())
    })
}

// INFERENCE MODEL
pub struct TractInferenceModel(tract_rs::InferenceModel);

/// Query an InferenceModel input counts.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_input_count(
    model: *const TractInferenceModel,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, inputs);
        let model = &(*model).0;
        *inputs = model.input_count()?;
        Ok(())
    })
}

/// Query an InferenceModel output counts.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_output_count(
    model: *const TractInferenceModel,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, outputs);
        let model = &(*model).0;
        *outputs = model.output_count()?;
        Ok(())
    })
}

/// Query the name of a model input.
///
/// The returned name must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_input_name(
    model: *const TractInferenceModel,
    input: usize,
    name: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name);
        *name = std::ptr::null_mut();
        let m = &(*model).0;
        *name = CString::new(&*m.input_name(input)?)?.into_raw();
        Ok(())
    })
}

/// Query the name of a model output.
///
/// The returned name must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_output_name(
    model: *const TractInferenceModel,
    output: usize,
    name: *mut *mut i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name);
        *name = std::ptr::null_mut();
        let m = &(*model).0;
        *name = CString::new(&*m.output_name(output)?)?.into_raw() as _;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_input_fact(
    model: *const TractInferenceModel,
    input_id: usize,
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, fact);
        *fact = std::ptr::null_mut();
        let f = (*model).0.input_fact(input_id)?;
        *fact = Box::into_raw(Box::new(TractInferenceFact(f)));
        Ok(())
    })
}

/// Set an input fact of an InferenceModel.
///
/// The `fact` argument is only borrowed by this function, it still must be destroyed.
/// `fact` can be set to NULL to erase the current output fact of the model.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_set_input_fact(
    model: *mut TractInferenceModel,
    input_id: usize,
    fact: *const TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        let f = fact.as_ref().map(|f| &f.0).cloned().unwrap_or_default();
        (*model).0.set_input_fact(input_id, f)?;
        Ok(())
    })
}

/// Change the model outputs nodes (by name).
///
/// `names` is an array containing `len` pointers to null terminated strings.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_set_output_names(
    model: *mut TractInferenceModel,
    len: usize,
    names: *const *const c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, names, *names);
        let node_names = (0..len)
            .map(|i| Ok(CStr::from_ptr(*names.add(i)).to_str()?.to_owned()))
            .collect::<Result<Vec<_>>>()?;
        (*model).0.set_output_names(&node_names)?;
        Ok(())
    })
}

/// Query an output fact for an InferenceModel.
///
/// The return model must be freed using `tract_inference_fact_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_output_fact(
    model: *const TractInferenceModel,
    output_id: usize,
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, fact);
        *fact = std::ptr::null_mut();
        let f = (*model).0.output_fact(output_id)?;
        *fact = Box::into_raw(Box::new(TractInferenceFact(f)));
        Ok(())
    })
}

/// Set an output fact of an InferenceModel.
///
/// The `fact` argument is only borrowed by this function, it still must be destroyed.
/// `fact` can be set to NULL to erase the current output fact of the model.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_set_output_fact(
    model: *mut TractInferenceModel,
    output_id: usize,
    fact: *const TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        let f = fact.as_ref().map(|f| &f.0).cloned().unwrap_or_default();
        (*model).0.set_output_fact(output_id, f)?;
        Ok(())
    })
}

/// Analyse an InferencedModel in-place.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_analyse(
    model: *mut TractInferenceModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        (*model).0.analyse()?;
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
        check_not_null!(model, *model, optimized);
        *optimized = std::ptr::null_mut();
        let m = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        let result = m.0.into_optimized()?;
        *optimized = Box::into_raw(Box::new(TractModel(result))) as _;
        Ok(())
    })
}

/// Transform a fully analysed InferenceModel to a TypedModel.
///
/// This function takes ownership of the InferenceModel `model` whether it succeeds
/// or not. `tract_inference_model_destroy` must not be used on `model`.
///
/// On the other hand, caller will be owning the newly created `optimized` model.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_into_typed(
    model: *mut *mut TractInferenceModel,
    typed: *mut *mut TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, *model, typed);
        *typed = std::ptr::null_mut();
        let m = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        let result = m.0.into_typed()?;
        *typed = Box::into_raw(Box::new(TractModel(result))) as _;
        Ok(())
    })
}

/// Destroy an InferenceModel.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_destroy(
    model: *mut *mut TractInferenceModel,
) -> TRACT_RESULT {
    release!(model)
}
// TYPED MODEL

pub struct TractModel(tract_rs::Model);

/// Query an InferenceModel input counts.
#[no_mangle]
pub unsafe extern "C" fn tract_model_input_count(
    model: *const TractModel,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, inputs);
        let model = &(*model).0;
        *inputs = model.input_count()?;
        Ok(())
    })
}

/// Query an InferenceModel output counts.
#[no_mangle]
pub unsafe extern "C" fn tract_model_output_count(
    model: *const TractModel,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, outputs);
        let model = &(*model).0;
        *outputs = model.output_count()?;
        Ok(())
    })
}

/// Query the name of a model input.
///
/// The returned name must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_model_input_name(
    model: *const TractModel,
    input: usize,
    name: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name);
        *name = std::ptr::null_mut();
        let m = &(*model).0;
        *name = CString::new(m.input_name(input)?)?.into_raw();
        Ok(())
    })
}

/// Query the input fact of a model.
///
/// Thre returned fact must be freed with tract_fact_destroy.
#[no_mangle]
pub unsafe extern "C" fn tract_model_input_fact(
    model: *const TractModel,
    input_id: usize,
    fact: *mut *mut TractFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, fact);
        *fact = std::ptr::null_mut();
        let f = (*model).0.input_fact(input_id)?;
        *fact = Box::into_raw(Box::new(TractFact(f)));
        Ok(())
    })
}

/// Query the name of a model output.
///
/// The returned name must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_model_output_name(
    model: *const TractModel,
    output: usize,
    name: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name);
        *name = std::ptr::null_mut();
        let m = &(*model).0;
        *name = CString::new(m.output_name(output)?)?.into_raw();
        Ok(())
    })
}

/// Query the output fact of a model.
///
/// Thre returned fact must be freed with tract_fact_destroy.
#[no_mangle]
pub unsafe extern "C" fn tract_model_output_fact(
    model: *const TractModel,
    input_id: usize,
    fact: *mut *mut TractFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, fact);
        *fact = std::ptr::null_mut();
        let f = (*model).0.output_fact(input_id)?;
        *fact = Box::into_raw(Box::new(TractFact(f)));
        Ok(())
    })
}

/// Change the model outputs nodes (by name).
///
/// `names` is an array containing `len` pointers to null terminated strings.
#[no_mangle]
pub unsafe extern "C" fn tract_model_set_output_names(
    model: *mut TractModel,
    len: usize,
    names: *const *const c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, names, *names);
        let node_names = (0..len)
            .map(|i| Ok(CStr::from_ptr(*names.add(i)).to_str()?.to_owned()))
            .collect::<Result<Vec<_>>>()?;
        (*model).0.set_output_names(&node_names)
    })
}

/// Give value one or more symbols used in the model.
///
/// * symbols is an array of `nb_symbols` pointers to null-terminated UTF-8 string for the symbols
///   names to substitue
/// * values is an array of `nb_symbols` integer values
#[no_mangle]
pub unsafe extern "C" fn tract_model_concretize_symbols(
    model: *mut TractModel,
    nb_symbols: usize,
    symbols: *const *const i8,
    values: *const i64,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, symbols, values);
        let model = &mut (*model).0;
        let mut table = vec![];
        for i in 0..nb_symbols {
            let name = CStr::from_ptr(*symbols.add(i) as _)
                .to_str()
                .with_context(|| {
                    format!("failed to parse symbol name for {i}th symbol (not utf8)")
                })?
                .to_owned();
            table.push((name, *values.add(i)));
        }
        model.concretize_symbols(table)
    })
}

/// Pulsify the model
///
/// * stream_symbol is the name of the stream symbol
/// * pulse expression is a dim to use as the pulse size (like "8", "P" or "3*p").
#[no_mangle]
pub unsafe extern "C" fn tract_model_pulse_simple(
    model: *mut *mut TractModel,
    stream_symbol: *const i8,
    pulse_expr: *const i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, *model, stream_symbol, pulse_expr);
        let model = &mut (**model).0;
        let stream_sym = CStr::from_ptr(stream_symbol as _)
            .to_str()
            .context("failed to parse stream symbol name (not utf8)")?;
        let pulse_dim = CStr::from_ptr(pulse_expr as _)
            .to_str()
            .context("failed to parse stream symbol name (not utf8)")?;
        model.pulse(stream_sym, pulse_dim)
    })
}

/// Apply a transform to the model.
#[no_mangle]
pub unsafe extern "C" fn tract_model_transform(
    model: *mut TractModel,
    transform: *const i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, transform);
        let t = CStr::from_ptr(transform as _)
            .to_str()
            .context("failed to parse transform name (not utf8)")?;
        (*model).0.transform(t)
    })
}

/// Declutter a TypedModel in-place.
#[no_mangle]
pub unsafe extern "C" fn tract_model_declutter(model: *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        (*model).0.declutter()
    })
}

/// Optimize a TypedModel in-place.
#[no_mangle]
pub unsafe extern "C" fn tract_model_optimize(model: *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        (*model).0.optimize()
    })
}

/// Perform a profile of the model using the provided inputs.
#[no_mangle]
pub unsafe extern "C" fn tract_model_profile_json(
    model: *mut TractModel,
    inputs: *mut *mut TractValue,
    json: *mut *mut i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, json);
        let input: Option<Vec<Value>> = if !inputs.is_null() {
            let input_len = (*model).0.input_count()?;
            Some(
                std::slice::from_raw_parts(inputs, input_len)
                    .iter()
                    .map(|tv| (**tv).0.clone())
                    .collect(),
            )
        } else {
            None
        };
        let profile = (*model).0.profile_json(input)?;
        *json = CString::new(profile)?.into_raw() as _;
        Ok(())
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
        check_not_null!(model, runnable);
        let m = Box::from_raw(*model).0;
        *model = std::ptr::null_mut();
        *runnable = Box::into_raw(Box::new(TractRunnable(m.into_runnable()?))) as _;
        Ok(())
    })
}

/// Query the number of properties in a model.
#[no_mangle]
pub unsafe extern "C" fn tract_model_property_count(
    model: *const TractModel,
    count: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, count);
        *count = (*model).0.property_keys()?.len();
        Ok(())
    })
}

/// Query the properties names of a model.
///
/// The "names" array should be big enough to fit `tract_model_property_count` string pointers.
///
/// Each name will have to be freed using `tract_free_cstring`.
#[no_mangle]
pub unsafe extern "C" fn tract_model_property_names(
    model: *const TractModel,
    names: *mut *mut i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, names);
        for (ix, name) in (*model).0.property_keys()?.iter().enumerate() {
            *names.add(ix) = CString::new(&**name)?.into_raw() as _;
        }
        Ok(())
    })
}

/// Query a property value in a model.
#[no_mangle]
pub unsafe extern "C" fn tract_model_property(
    model: *const TractModel,
    name: *const i8,
    value: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name, value);
        let name = CStr::from_ptr(name as _)
            .to_str()
            .context("failed to parse property name (not utf8)")?
            .to_owned();
        let v = (*model).0.property(name).context("Property not found")?;
        *value = Box::into_raw(Box::new(TractValue(v)));
        Ok(())
    })
}

/// Destroy a TypedModel.
#[no_mangle]
pub unsafe extern "C" fn tract_model_destroy(model: *mut *mut TractModel) -> TRACT_RESULT {
    release!(model)
}

// RUNNABLE MODEL
pub struct TractRunnable(tract_rs::Runnable);

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
        check_not_null!(runnable, state);
        *state = std::ptr::null_mut();
        let s = (*runnable).0.spawn_state()?;
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
        check_not_null!(runnable);
        let mut s = (*runnable).0.spawn_state()?;
        state_run(&mut s, inputs, outputs)
    })
}

/// Query a Runnable input counts.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_input_count(
    model: *const TractRunnable,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, inputs);
        let model = &(*model).0;
        *inputs = model.input_count()?;
        Ok(())
    })
}

/// Query an Runnable output counts.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_output_count(
    model: *const TractRunnable,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, outputs);
        let model = &(*model).0;
        *outputs = model.output_count()?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_runnable_release(runnable: *mut *mut TractRunnable) -> TRACT_RESULT {
    release!(runnable)
}

// VALUE
pub struct TractValue(tract_rs::Value);

/// Create a TractValue (aka tensor) from caller data and metadata.
///
/// This call copies the data into tract space. All the pointers only need to be alive for the
/// duration of the call.
///
/// rank is the number of dimensions of the tensor (i.e. the length of the shape vector).
///
/// The returned value must be destroyed by `tract_value_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_value_from_bytes(
    datum_type: DatumType,
    rank: usize,
    shape: *const usize,
    data: *mut c_void,
    value: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(value);
        *value = std::ptr::null_mut();
        let shape = std::slice::from_raw_parts(shape, rank);
        let len = shape.iter().product::<usize>();
        let data = std::slice::from_raw_parts(data as *const u8, len * datum_type.size_of());
        let it = Value::from_bytes(datum_type, shape, data)?;
        *value = Box::into_raw(Box::new(TractValue(it)));
        Ok(())
    })
}

/// Destroy a value.
#[no_mangle]
pub unsafe extern "C" fn tract_value_destroy(value: *mut *mut TractValue) -> TRACT_RESULT {
    release!(value)
}

/// Inspect part of a value. Except `value`, all argument pointers can be null if only some specific bits
/// are required.
#[no_mangle]
pub unsafe extern "C" fn tract_value_as_bytes(
    value: *mut TractValue,
    datum_type: *mut DatumType,
    rank: *mut usize,
    shape: *mut *const usize,
    data: *mut *const c_void,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(value);
        let value = &(*value).0;
        let bits = value.as_bytes()?;
        if !datum_type.is_null() {
            *datum_type = bits.0;
        }
        if !rank.is_null() {
            *rank = bits.1.len();
        }
        if !shape.is_null() {
            *shape = bits.1.as_ptr();
        }
        if !data.is_null() {
            *data = bits.2.as_ptr() as _;
        }
        Ok(())
    })
}

// STATE
pub struct TractState(tract_rs::State);

/// Run a turn on a model state
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
        check_not_null!(state, inputs, outputs);
        state_run(&mut (*state).0, inputs, outputs)
    })
}

/// Query a State input counts.
#[no_mangle]
pub unsafe extern "C" fn tract_state_input_count(
    state: *const TractState,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(state, inputs);
        let state = &(*state).0;
        *inputs = state.input_count()?;
        Ok(())
    })
}

/// Query an State output counts.
#[no_mangle]
pub unsafe extern "C" fn tract_state_output_count(
    state: *const TractState,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(state, outputs);
        let state = &(*state).0;
        *outputs = state.output_count()?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_state_destroy(state: *mut *mut TractState) -> TRACT_RESULT {
    release!(state)
}

// FACT
pub struct TractFact(tract_rs::Fact);

/// Parse a fact specification string into an Fact.
///
/// The returned fact must be free with `tract_fact_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_fact_parse(
    model: *mut TractModel,
    spec: *const c_char,
    fact: *mut *mut TractFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, spec, fact);
        let spec = CStr::from_ptr(spec).to_str()?;
        let f: tract_rs::Fact = spec.as_fact(&mut (*model).0)?.as_ref().clone();
        *fact = Box::into_raw(Box::new(TractFact(f)));
        Ok(())
    })
}

/// Write a fact as its specification string.
///
/// The returned string must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_fact_dump(
    fact: *const TractFact,
    spec: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(fact, spec);
        *spec = CString::new(format!("{}", (*fact).0))?.into_raw();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_fact_destroy(fact: *mut *mut TractFact) -> TRACT_RESULT {
    release!(fact)
}

// INFERENCE FACT
pub struct TractInferenceFact(tract_rs::InferenceFact);

/// Parse a fact specification string into an InferenceFact.
///
/// The returned fact must be free with `tract_inference_fact_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_parse(
    model: *mut TractInferenceModel,
    spec: *const c_char,
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, spec, fact);
        let spec = CStr::from_ptr(spec).to_str()?;
        let f: tract_rs::InferenceFact = spec.as_fact(&mut (*model).0)?.as_ref().clone();
        *fact = Box::into_raw(Box::new(TractInferenceFact(f)));
        Ok(())
    })
}

/// Creates an empty inference fact.
///
/// The returned fact must be freed by the caller using tract_inference_fact_destroy
#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_empty(
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(fact);
        *fact = Box::into_raw(Box::new(TractInferenceFact(Default::default())));
        Ok(())
    })
}

/// Write an inference fact as its specification string.
///
/// The returned string must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_dump(
    fact: *const TractInferenceFact,
    spec: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(fact, spec);
        *spec = CString::new(format!("{}", (*fact).0))?.into_raw();
        Ok(())
    })
}

/// Destroy a fact.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_destroy(
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    release!(fact)
}

// MISC

// HELPERS

unsafe fn state_run(
    state: &mut State,
    inputs: *mut *mut TractValue,
    outputs: *mut *mut TractValue,
) -> Result<()> {
    let values: Vec<_> = std::slice::from_raw_parts(inputs, state.input_count()?)
        .iter()
        .map(|tv| (**tv).0.clone())
        .collect();
    let values = state.run(values)?;
    for (i, value) in values.into_iter().enumerate() {
        *(outputs.add(i)) = Box::into_raw(Box::new(TractValue(value)))
    }
    Ok(())
}
