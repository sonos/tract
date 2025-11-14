use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaFunction, LaunchArgs, LaunchConfig, PushKernelArg};
use num_traits::One;
use std::fmt;
use tract_core::internal::*;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{LibraryName, WARP_SIZE, get_cuda_view, launch_args};

#[derive(Debug, Clone)]
pub struct CudaFlashAttn;

impl fmt::Display for CudaFlashAttn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl CudaFlashAttn {
    pub fn name(&self) -> Cow<'_, str> {
        format!("{self}").into()
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

    #[allow(clippy::too_many_arguments)]
    pub fn eval(
        &self,
        stream: &TractCudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: Option<&DeviceTensor>,
        scale: f32,
        is_causal: bool,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe {
            DeviceTensor::uninitialized_dt(
                q.datum_type(),
                &self.output_shape(q.shape(), k.shape(), v.shape())?,
            )?
        };

        self.dispatch_eval(stream, q, k, v, mask, scale, &output, is_causal)?;
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
        mask: Option<&DeviceTensor>,
        scale: f32,
        out: &DeviceTensor,
        is_causal: bool,
    ) -> TractResult<()> {
        ensure!(q.datum_type() == DatumType::F16 && q.datum_type() == out.datum_type());
        ensure!(k.datum_type() == DatumType::F16 && k.datum_type() == v.datum_type());

        ensure!(out.shape() == self.output_shape(q.shape(), k.shape(), v.shape())?.as_slice());
        ensure!(!is_causal || mask.is_none());
        ensure!(mask.is_none_or(|m| m.datum_type() == DatumType::F16));

        let ctxt = cuda_context();
        let q_shape = q.shape();

        let b = q_shape[0];
        let n_qh = q_shape[1];
        let len_q = q_shape[2];
        let len_kv = k.shape()[2];
        let d = q_shape[3];

        ensure!(n_qh % k.shape()[1] == 0);
        ensure!(k.shape()[0] == b);

        let head_ratio = n_qh / k.shape()[1];
        let block_q = 64;
        let block_kv = 32;

        let n_warps = 4;

        let num_full_q_blocks = len_q / block_q;
        let tb_size = n_warps * WARP_SIZE;
        let smem_size = block_q.max(block_kv * 3) * d * size_of::<f16>();

        let use_mask = mask.is_some();

        let null_ptr = stream.null::<u8>()?;

        let q_view = get_cuda_view(q);
        let k_view = get_cuda_view(k);
        let v_view = get_cuda_view(v);
        let m_view = mask.map(get_cuda_view).unwrap_or_else(|| null_ptr.as_view());
        let o_view = get_cuda_view(out);

        let kernel_launcher = |suffix: &str, num_q_blocks: usize| -> TractResult<()> {
            let func = ctxt.load_pipeline(
                LibraryName::FlashAttn,
                format!("attention_v5_{suffix}{block_q}_{block_kv}_{d}_{is_causal}_{use_mask}"),
            )?;

            func.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_size as _,
            )?;

            let mut launch_args = stream.launch_builder(&func);
            launch_args.arg(&q_view);
            launch_args.arg(&k_view);
            launch_args.arg(&v_view);
            launch_args.arg(&m_view);
            launch_args.arg(&o_view);
            launch_args.arg(&b);
            launch_args.arg(&n_qh);
            launch_args.arg(&head_ratio);
            launch_args.arg(&len_q);
            launch_args.arg(&k.shape()[2]);
            launch_args.arg(&scale);

            let cfg = LaunchConfig {
                grid_dim: (num_q_blocks as _, n_qh as _, b as _),
                block_dim: (tb_size as _, 1, 1),
                shared_mem_bytes: smem_size as _,
            };
            unsafe {
                launch_args.launch(cfg);
            }
            Ok(())
        };

        if num_full_q_blocks > 0 {
            let mut str = "full_".to_string();
            if len_kv % block_kv != 0 {
                str.push_str("kv_rem_");
            }
            kernel_launcher(&str, num_full_q_blocks)?;
        }

        if len_q % block_q != 0 {
            let mut str = "tail_".to_string();
            if len_kv % block_kv != 0 {
                str.push_str("kv_rem_");
            }
            kernel_launcher(&str, 1)?;
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
        is_causal: bool,
        create_mask: bool,
    ) -> TractResult<()> {
        ensure!(!(create_mask && is_causal));
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

            let m = if create_mask {
                Tensor::from_shape(
                    &m_shape,
                    &(0..m_len).map(|f| f16::from_f32(1f32)).collect::<Vec<_>>(),
                )?
            } else {
                tensor0(0.0f32) // Unused 
            };

            let cuda_m = m.clone().into_device()?;
            let cuda_output = CudaFlashAttn.eval(
                stream,
                &q.clone().into_device()?,
                &k.clone().into_device()?,
                &v.clone().into_device()?,
                if create_mask { Some(&cuda_m) } else { None },
                scale,
                is_causal,
            )?;

            let mut ref_inputs = tvec!(q.into(), k.into(), v.into());

            if create_mask {
                ref_inputs.push(m.into())
            };
            let ref_output = Sdpa {
                scale: Some(scale.into()),
                datum_type: DatumType::F16,
                acc_datum_type: DatumType::F32,
                is_causal,
            }
            .eval(ref_inputs)?;

            cuda_output.to_host()?.close_enough(&ref_output[0], Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_nernst_fattn() -> TractResult<()> {
        run_test_case(1, 1, 1, 64, 1, 128, 1.0f32, false, false)?;
        run_test_case(1, 2, 1, 64, 1, 128, 1.0f32, false, true)?;
        run_test_case(1, 2, 2, 0, 1, 64, 1.0f32, false, false)?;
        run_test_case(2, 4, 2, 123, 1, 64, 1.0f32, false, false)?;
        run_test_case(1, 2, 2, 0, 1, 64, 1.0f32, true, false)?;
        run_test_case(1, 1, 1, 64, 64, 128, 1.0f32, false, false)?;
        run_test_case(2, 32, 4, 64, 64, 128, 1.0f32, false, false)?;
        run_test_case(1, 1, 1, 64, 64, 128, 1.0f32, false, true)?;
        run_test_case(1, 1, 1, 64, 64, 128, 1.0f32, true, false)?;
        Ok(())
    }
}
