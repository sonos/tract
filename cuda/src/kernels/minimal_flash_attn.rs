use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaFunction, LaunchArgs, LaunchConfig, PushKernelArg};
use num_traits::One;
use std::f32::INFINITY;
use std::fmt;
use tract_core::internal::*;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{LibraryName, WARP_SIZE, get_cuda_view, launch_args};

const CUDA_CC_TURING: i32 = 750;
const CUDA_CC_AMPERE: i32 = 800;
const FATTN_KQ_STRIDE: usize = 256;

#[derive(Debug, Clone)]
pub struct MinimalFlashAttn;

impl fmt::Display for MinimalFlashAttn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl MinimalFlashAttn {
    pub fn is_supported_dts(q_dt: DatumType, k_dt: DatumType, v_dt: DatumType, m_dt: DatumType) -> bool {
        k_dt == DatumType::F16 && (k_dt == v_dt) && (k_dt == q_dt) && (k_dt == m_dt)
    }

    pub fn name(&self) -> Cow<'_, str> {
        format!("{self}").into()
    }

    pub fn kernel_name(
        &self,
        d: usize,
        q_dt: DatumType,
        k_dt: DatumType,
        v_dt: DatumType,
        m_dt: DatumType,
    ) -> TractResult<String> {
        ensure!(
            Self::is_supported_dts(q_dt, k_dt, v_dt, m_dt),
            "Unsupported dts K: {:?} V: {:?} for Cuda Flash Attention Op",
            k_dt,
            v_dt
        );
        Ok(format!(
            "attention_v5_{d}"
        ))
    }

    pub fn output_shape<D: DimLike + One>(
        &self,
        q: &[D],
        k: &[D],
        v: &[D],
    ) -> TractResult<TVec<D>> {
        ensure!(q.len() == 4, "Q rank must be 4 (got {})", q.len());
        ensure!(
            k.len() == q.len() && v.len() == q.len(),
            "K and V must have the same rank as Q (Q={}, K={}, V={})",
            q.len(),
            k.len(),
            v.len()
        );

        match (q, k, v) {
            ([b, qh, s, _], [_, kh, _, _], [_, vh, _, d]) => {
                let (qh_i, kh_i, vh_i) = (qh.to_i64()?, kh.to_i64()?, vh.to_i64()?);
                ensure!(kh_i == vh_i, "K and V heads mismatch (K={}, V={})", kh_i, vh_i);
                ensure!(kh_i > 0, "K/V heads must be > 0 (got {kh_i})");
                ensure!(
                    qh_i % kh_i == 0,
                    "Q heads ({qh_i}) must be a multiple of K/V heads ({kh_i})"
                );
                Ok(tvec![b.clone(), qh.clone(), s.clone(), d.clone()])
            }
            _ => bail!("Inconsistent shapes: expected [B,H,S,D] for Q/K/V."),
        }
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: &DeviceTensor,
        scale: f32,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe {
            DeviceTensor::uninitialized_dt(
                q.datum_type(),
                &self.output_shape(q.shape(), k.shape(), v.shape())?,
            )?
        };

        self.dispatch_eval(stream, q, k, v, mask, scale, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: &DeviceTensor,
        scale: f32,
        out: &DeviceTensor,
    ) -> TractResult<()> {
        let ctxt = cuda_context();
        ensure!(q.datum_type() == DatumType::F16 && q.datum_type() == out.datum_type());
        ensure!(out.shape() == self.output_shape(q.shape(), k.shape(), v.shape())?.as_slice());

        let q_shape = q.shape();

        let b = q_shape[0] * q_shape[1];
        //let nh = q_shape[1];
        let len_q = q_shape[2];
        let d = q_shape[3];
        ensure!(q_shape[1] == k.shape()[1]);
        let block_q = 64;
        let block_kv= 64;
        let n_warps = 4;

        let num_blocks = b * len_q.div_ceil(block_q);
        let tb_size = n_warps * WARP_SIZE;
        let smem_size = block_q.max(block_kv * 3) * d * size_of::<f16>();

        let func = ctxt.load_pipeline(LibraryName::MinimalFlashAttn, self.kernel_name(d, q.datum_type(), k.datum_type(), v.datum_type(), mask.datum_type())?)?;
        func.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size as _)?;
        
        let q_view = get_cuda_view(q);
        let k_view = get_cuda_view(k);
        let v_view = get_cuda_view(v);
        let o_view = get_cuda_view(out);

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&q_view);
        launch_args.arg(&k_view);
        launch_args.arg(&v_view);
        launch_args.arg(&o_view);
        launch_args.arg(&b);
        launch_args.arg(&len_q);
        launch_args.arg(&k.shape()[2]);

        let cfg = LaunchConfig { grid_dim: (num_blocks as _, 1, 1), block_dim: (tb_size as _, 1, 1), shared_mem_bytes: smem_size as _};
        unsafe {
            launch_args.launch(cfg);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Float;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::sdpa::Sdpa;

    use crate::context::CUDA_STREAM;

    use super::*;

    fn run_test_case(
        batch: usize,
        q_heads: usize,
        kv_heads: usize,
        past_seq_len: usize,
        seq_len: usize,
        out_dim: usize,
        scale: f32,
    ) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let q_shape = [batch, q_heads, seq_len, out_dim];
            let kv_shape = [batch, kv_heads, past_seq_len + seq_len, out_dim];
            let m_shape = [1, 1, seq_len, past_seq_len + seq_len];

            let q_len = q_shape.iter().product::<usize>();
            let kv_len = kv_shape.iter().product::<usize>();
            let m_len = m_shape.iter().product::<usize>();

            let q = Tensor::from_shape(
                &q_shape,
                &(0..q_len).map(|f| f16::from_f32(f as f32 / q_len as f32)).collect::<Vec<_>>(),
            )?;

            let k = Tensor::from_shape(
                &kv_shape,
                &(0..kv_len).map(|f| f16::from_f32(f as f32 / kv_len as f32)).collect::<Vec<_>>(),
            )?;

            let v = Tensor::from_shape(
                &kv_shape,
                &(0..kv_len).map(|f| f16::from_f32(f as f32 / kv_len as f32)).collect::<Vec<_>>(),
            )?;

            let m = Tensor::from_shape(
                &m_shape,
                &(0..m_len).map(|f| f16::from_f32(1f32)).collect::<Vec<_>>(),
            )?;

            let cuda_output = MinimalFlashAttn.eval(
                stream,
                &q.clone().into_device()?,
                &k.clone().into_device()?,
                &v.clone().into_device()?,
                &m.clone().into_device()?,
                scale,
            )?;

            let ref_output = Sdpa {
                scale: Some(scale.into()),
                datum_type: DatumType::F16,
                acc_datum_type: DatumType::F32,
                is_causal: false,
            }
            .eval(tvec!(q.into(), k.into(), v.into(), m.into()))?;

            cuda_output.to_host()?.close_enough(&ref_output[0], Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_fattn_mma_f16() -> TractResult<()> {
        run_test_case(1, 1, 1, 0, 64, 64, 1.0f32)?;
        run_test_case(1, 1, 1, 0, 256, 64, 1.0f32)?;
        run_test_case(1, 1, 1, 0, 256, 128, 1.0f32)?;
        run_test_case(1, 4, 4, 256, 256, 64, 1.0f32)?;
        run_test_case(1, 8, 8, 512, 512, 128, 1.0f32)?;
        run_test_case(1, 1, 1, 4096, 4096, 64, 1.0f32)?;
        run_test_case(1, 4, 4, 256, 2048, 64, 1.0f32)?;
        run_test_case(1, 32, 32, 512, 1024, 128, 1.0f32)?;
        //run_test_case(1, 8, 8, 4096, 4096, 128, 1.0f32)?;
        //run_test_case(1, 8, 8, 1, 1, 80, 1.0f32)?;
        //run_test_case(2, 4, 2, 1, 1, 128, 1.0f32)?;
        //run_test_case(2, 8, 8, 0, 1, 96, 1.0f32)?;
        //run_test_case(1, 1, 1, 3, 2, 256, 1.0f32)?;
        //run_test_case(2, 2, 1, 3, 1, 112, 1.0f32)?;
        Ok(())
    }
}
