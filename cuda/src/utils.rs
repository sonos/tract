use std::sync::OnceLock;

use tract_core::internal::tract_smallvec::ToSmallVec;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::*;
use tract_gpu::tensor::DeviceTensor;

use crate::Q40_ROW_PADDING;
use crate::ops::GgmlQuantQ81Fact;

static CULIBS_PRESENT: OnceLock<bool> = OnceLock::new();

pub fn are_culibs_present() -> bool {
    *CULIBS_PRESENT.get_or_init(|| unsafe {
        cudarc::driver::sys::is_culib_present()
            && cudarc::runtime::sys::is_culib_present()
            && cudarc::nvrtc::sys::is_culib_present()
            && cudarc::cublas::sys::is_culib_present()
    })
}

pub fn get_ggml_q81_fact(t: &DeviceTensor) -> Option<GgmlQuantQ81Fact> {
    if let DeviceTensor::Owned(t) = t {
        t.opaque_fact().and_then(|of| of.downcast_ref::<GgmlQuantQ81Fact>()).cloned()
    } else if let DeviceTensor::ArenaView(t) = t {
        t.opaque_fact().and_then(|of| of.downcast_ref::<GgmlQuantQ81Fact>()).cloned()
    } else {
        None
    }
}

pub fn pad_q40(q40_bqv: &BlobWithFact) -> TractResult<BlobWithFact> {
    let q40_bqf =
        q40_bqv.fact.downcast_ref::<BlockQuantFact>().context("Expected BlockQuantFact")?;
    let shape = q40_bqf.shape();
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

    Ok(BlobWithFact {
        fact: Box::new(BlockQuantFact::new(q40_bqf.format.clone(), new_shape)),
        value: Arc::new(Blob::from_bytes(&new_data)?),
    })
}
