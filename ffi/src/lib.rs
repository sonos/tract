use std::{ffi::CStr, os::raw::c_char};

use tract_nnef::prelude::Framework;

pub struct Nnef(tract_nnef::internal::Nnef);

#[no_mangle]
pub extern "C" fn tract_nnef_create(nnef: *mut *mut Nnef) {
    unsafe {
        *nnef = Box::into_raw(Box::new(Nnef(tract_nnef::nnef())));
    }
}

#[no_mangle]
pub extern "C" fn tract_nnef_destroy(nnef: *mut Nnef) {
    unsafe { Box::from_raw(nnef) };
}

pub struct TypedModel(tract_nnef::internal::TypedModel);

#[no_mangle]
pub extern "C" fn tract_nnef_model_for_path(
    nnef: &Nnef,
    path: *const c_char,
    model: *mut *mut TypedModel,
) {
    unsafe {
        let path = CStr::from_ptr(path).to_str().unwrap();
        let m = Box::new(TypedModel(nnef.0.model_for_path(path).unwrap()));
        *model = Box::into_raw(m)
    };
}

#[no_mangle]
pub extern "C" fn tract_typed_model_destroy(model: *mut TypedModel) {
    unsafe { Box::from_raw(model) };
}
