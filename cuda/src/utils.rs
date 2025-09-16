use cudarc::driver::sys::Lib;
use tract_core::internal::tract_smallvec::ToSmallVec;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::*;
use tract_gpu::tensor::DeviceTensor;

use crate::Q40_ROW_PADDING;
use crate::ops::GgmlQuantQ81Fact;
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

pub fn get_quant_fact(t: &DeviceTensor, format: &dyn BlockQuant) -> Option<BlockQuantFact> {
    if let DeviceTensor::Owned(t) = t {
        t.downcast_ref::<CudaTensor>()
            .expect("Non Cuda Tensor in Cuda context")
            .opaque_fact()
            .map(|of| of.downcast_ref::<BlockQuantFact>().unwrap().clone())
            .filter(|bqf| bqf.format.same_as(format))
    } else {
        None
    }
}

pub fn get_ggml_q81_fact(t: &DeviceTensor) -> Option<GgmlQuantQ81Fact> {
    if let DeviceTensor::Owned(t) = t {
        t.downcast_ref::<CudaTensor>()
            .expect("Non Cuda Tensor in Cuda context")
            .opaque_fact()
            .map(|of| of.downcast_ref::<GgmlQuantQ81Fact>().unwrap().clone())
    } else if let DeviceTensor::ArenaView(t) = t {
        t.opaque_fact().map(|of| of.downcast_ref::<GgmlQuantQ81Fact>().unwrap().clone())
    } else {
        None
    }
}

pub fn pad_q40(q40_bqv: &BlockQuantValue) -> TractResult<BlockQuantValue> {
    let shape = q40_bqv.fact.shape();
    ensure!(shape.len() >= 2);

    let k = *shape.last().unwrap();
    ensure!(k % 32 == 0);

    let to_pad = k.next_multiple_of(Q40_ROW_PADDING) - k;
    if to_pad == 0 {
        return Ok(q40_bqv.clone()); // No padding needed
    }

    let outer_rows: usize = shape[..shape.len() - 1].iter().product();
    let row_bytes = k * Q4_0.block_bytes() / Q4_0.block_len();

    let pad_quant = Q4_0.quant_f32(&vec![0f32; to_pad])?;
    let pad_bytes = pad_quant.len();

    let mut new_data = Vec::with_capacity(outer_rows * (row_bytes + pad_bytes));
    let old_bytes = q40_bqv.value.as_bytes();

    for row in 0..outer_rows {
        let start = row * row_bytes;
        new_data.extend_from_slice(&old_bytes[start..start + row_bytes]);
        new_data.extend_from_slice(&pad_quant);
    }
    let mut new_shape = shape.to_smallvec();
    *new_shape.last_mut().unwrap() += to_pad;

    Ok(BlockQuantValue {
        fact: BlockQuantFact::new(q40_bqv.fact.format.clone(), new_shape),
        value: Arc::new(Blob::from_bytes(&new_data)?),
    })
}
