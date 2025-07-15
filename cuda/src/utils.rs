use cudarc::driver::sys::Lib;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, Q4_0};
use tract_gpu::tensor::DeviceTensor;

use crate::tensor::CudaTensor;

// Code copied from Cudarc for checking Cuda presence
fn get_lib_name_candidates(lib_name: &str) -> Vec<String> {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

    let pointer_width = if cfg!(target_pointer_width = "32") {
        "32"
    } else if cfg!(target_pointer_width = "64") {
        "64"
    } else {
        panic!("Unsupported target pointer width")
    };

    let major = "12";
    let minor = "6";

    [
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{minor}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{minor}_0{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}0_{minor}{DLL_SUFFIX}"),
        // See issue #242
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_10{DLL_SUFFIX}"),
        // See issue #246
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}0_0{DLL_SUFFIX}"),
        // See issue #260
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_9{DLL_SUFFIX}"),
        // See issue #274
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.{major}"),
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.11"),
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.10"),
        // See issue #296
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.1"),
    ]
    .into()
}

pub fn get_cuda_lib() -> Option<Lib> {
    let lib_names = std::vec!["cuda", "nvcuda"];
    let choices: std::vec::Vec<_> =
        lib_names.iter().flat_map(|l| get_lib_name_candidates(l)).collect();
    unsafe {
        for choice in choices.iter() {
            if let Ok(lib) = Lib::new(choice) {
                return Some(lib);
            }
        }
        None
    }
}


pub fn get_q40_fact(t: &DeviceTensor) -> Option<BlockQuantFact> {
    if let DeviceTensor::Owned(t) = t {
        t.downcast_ref::<CudaTensor>()
        .expect("Non Cuda Tensor in Cuda context")
        .block_quant_fact().filter(|bqf| bqf.format.same_as(&Q4_0))
    } else {
        None
    }
}