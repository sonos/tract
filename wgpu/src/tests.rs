#![cfg(test)]

use tract_core::internal::*;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};

use crate::kernels::cast::wgpu_cast_dispatch;
use crate::kernels::copy::wgpu_copy_nd_dispatch;
use crate::with_wgpu_queue;

fn close(a: &Tensor, b: &Tensor) -> TractResult<()> {
    a.close_enough(b, Approximation::Close)
}

#[test]
fn upload_download_roundtrip() -> TractResult<()> {
    with_wgpu_queue(|_| {
        let t = Tensor::from_shape(&[2, 3], &(0..6).map(|i| i as f32).collect::<Vec<_>>())?;
        let d = t.clone().into_device()?;
        let back = d.to_host()?.into_tensor();
        assert_eq!(t, back);
        Ok(())
    })
}

#[test]
fn copy_nd_transposed_matches_cpu() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let t = Tensor::from_shape(&[2, 3], &(0..6).map(|i| i as f32).collect::<Vec<_>>())?;
        let src = t.clone().into_device()?;
        let dst = DeviceTensor::uninitialized_dt(DatumType::F32, &[3, 2])?;
        wgpu_copy_nd_dispatch(&src, 0, &[1, 3], &dst, 0, &[3, 2], &[2, 1])?;
        q.flush()?;
        close(&dst.to_host()?.into_tensor(), &t.clone().permute_axes(&[1, 0])?)
    })
}

#[test]
fn cast_f32_f16_roundtrip() -> TractResult<()> {
    with_wgpu_queue(|q| {
        if !q.context().shader_f16() {
            return Ok(());
        }
        let t = Tensor::from_shape(&[4], &[1.0f32, -2.5, 0.25, 100.0])?;
        let src = t.clone().into_device()?;
        let half = DeviceTensor::uninitialized_dt(DatumType::F16, &[4])?;
        let back = DeviceTensor::uninitialized_dt(DatumType::F32, &[4])?;
        wgpu_cast_dispatch(&src, &half)?;
        wgpu_cast_dispatch(&half, &back)?;
        q.flush()?;
        close(&back.to_host()?.into_tensor(), &t)
    })
}
