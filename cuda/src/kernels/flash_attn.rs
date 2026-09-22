use cudarc::driver::sys::cuTensorMapEncodeTiled;
use cudarc::driver::sys::{
    CUfunction_attribute, CUresult, CUtensorMap, CUtensorMapDataType, CUtensorMapFloatOOBfill,
    CUtensorMapInterleave, CUtensorMapL2promotion, CUtensorMapSwizzle,
};
use cudarc::driver::{
    CudaFunction, DevicePtr, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg,
};
use num_traits::One;
use std::fmt;
use tract_core::internal::*;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};

use crate::context::{TractCudaStream, cuda_context, tma_disabled_by_env};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{LibraryName, WARP_SIZE, get_cuda_view, launch_args};

/// 2-D TMA map for a row-major half array. `globalDim[0]` is the fastest
/// axis (head dim). `boxDim` is in elements, not bytes. `globalStrides` has
/// rank-1 entries (byte stride of dim 1).
fn make_tma_desc(
    global_addr: cudarc::driver::sys::CUdeviceptr,
    total_rows: usize,
    dim: usize,
    padded_dim: usize,
    block_kv: usize,
) -> TractResult<CUtensorMap> {
    let mut desc = CUtensorMap { opaque: [0u64; 16] };
    let global_dim: [u64; 2] = [dim as u64, total_rows as u64];
    let global_strides: [u64; 1] = [(dim * 2) as u64];
    let box_dim: [u32; 2] = [padded_dim as u32, block_kv as u32];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            &mut desc,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_addr as *mut std::ffi::c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    ensure!(
        result == CUresult::CUDA_SUCCESS,
        "cuTensorMapEncodeTiled failed (CUresult = {:?}): invalid tensor map parameters",
        result
    );
    Ok(desc)
}

#[repr(C)]
#[derive(Copy, Clone)]
struct TensorMapArg {
    opaque: [u64; 16],
}
unsafe impl DeviceRepr for TensorMapArg {}

fn tensor_map_arg(desc: CUtensorMap) -> TensorMapArg {
    TensorMapArg { opaque: desc.opaque }
}

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
        // PADDED_DIM — same rounding logic as the kernel instantiation
        let padded_d = d.div_ceil(64) * 64;
        let smem_size = block_q.max(block_kv * 3) * padded_d * size_of::<f16>();

        // RTX 5090 (SM120) only, and only when the padded row is 128 B so
        // TMA SWIZZLE_128B matches the software swizzle the MMA already uses.
        // Jetson / 4060 have no TMA; Hopper SM90 is a separate path.
        let sm_major = ctxt.properties().major;
        let use_tma = sm_major == 12 && padded_d == 64 && d == 64 && !tma_disabled_by_env();
        let smem_size = if use_tma { smem_size + 2 * size_of::<u64>() } else { smem_size };

        let mask_mode = if is_causal {
            "causal"
        } else if mask.is_some() {
            "mask"
        } else {
            "nomask"
        };

        let null_ptr = stream.null()?;

        let q_view = get_cuda_view(q);
        let k_view = get_cuda_view(k);
        let v_view = get_cuda_view(v);
        let m_view = mask.map(get_cuda_view).unwrap_or_else(|| null_ptr.as_view());
        let o_view = get_cuda_view(out);

        let mask_strides = if let Some(m) = mask {
            let strides = compute_broadcast_strides(m.shape(), m.strides())?;
            (strides[0], strides[1])
        } else {
            (0, 0)
        };

        // By-value 128-byte maps (grid-constant). Avoids a device allocation
        // that would be freed while the kernel is still running.
        let kv_heads = n_qh / head_ratio;
        let total_kv_rows = b * kv_heads * len_kv;
        let zero_map = TensorMapArg { opaque: [0u64; 16] };
        let k_tma = if use_tma {
            tensor_map_arg(make_tma_desc(
                k_view.device_ptr(stream).0,
                total_kv_rows,
                d,
                padded_d,
                block_kv,
            )?)
        } else {
            zero_map
        };
        let v_tma = if use_tma {
            tensor_map_arg(make_tma_desc(
                v_view.device_ptr(stream).0,
                total_kv_rows,
                d,
                padded_d,
                block_kv,
            )?)
        } else {
            zero_map
        };

        let kernel_launcher = |suffix: &str, num_q_blocks: usize| -> TractResult<()> {
            let func = ctxt.load_pipeline(
                LibraryName::FlashAttn,
                format!("attention_{suffix}{block_q}_{block_kv}_{d}_{mask_mode}"),
            )?;

            func.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_size as _,
            )?;

            let mut launch_args = TractLaunchArgs::new(stream, &func);
            launch_args.push_view(&q_view);
            launch_args.push_view(&k_view);
            launch_args.push_view(&v_view);
            launch_args.push_view(&m_view);
            launch_args.push_view(&o_view);
            launch_args.push_i32(b);
            launch_args.push_i32(n_qh);
            launch_args.push_i32(head_ratio);
            launch_args.push_i32(len_q);
            launch_args.push_i32(k.shape()[2]);
            launch_args.push_i32(mask_strides.0);
            launch_args.push_i32(mask_strides.1);
            launch_args.push::<f32>(scale);
            launch_args.push_copy(k_tma);
            launch_args.push_copy(v_tma);

            let cfg = LaunchConfig {
                grid_dim: (num_q_blocks as _, n_qh as _, b as _),
                block_dim: (tb_size as _, 1, 1),
                shared_mem_bytes: smem_size as _,
            };

            launch_args.launch(cfg)
        };

        if num_full_q_blocks > 0 {
            kernel_launcher("fullq_", num_full_q_blocks)?;
        }

        if !len_q.is_multiple_of(block_q) {
            kernel_launcher("tailq_", 1)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Float;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::sdpa::Sdpa;

    use super::*;

    #[allow(clippy::too_many_arguments)]
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
        crate::with_cuda_stream(|stream| {
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
            .eval(&EvalContext::out_of_plan(), ref_inputs)?;

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
        // d=64, seq_len=64 (= block_q) so fullq_ is launched. kv_len is a
        // multiple of BLOCK_KV (32), so every KV tile can take TMA on SM120.
        run_test_case(1, 2, 2, 0, 64, 64, 1.0f32, false, false)?;
        run_test_case(1, 2, 2, 0, 64, 64, 1.0f32, true, false)?;
        run_test_case(2, 4, 2, 0, 64, 64, 1.0f32, false, false)?;
        run_test_case(1, 1, 1, 0, 64, 64, 1.0f32, false, false)?;
        run_test_case(1, 2, 2, 0, 64, 64, 1.0f32, false, true)?;
        // kv_len = 96 (3 tiles)
        run_test_case(1, 2, 2, 32, 64, 64, 1.0f32, false, false)?;
        // kv_len = 128 (4 tiles)
        run_test_case(2, 4, 2, 64, 64, 64, 1.0f32, false, false)?;
        // kv_len = 50: one full TMA tile + 18-element cp.async.cg tail.
        // seq_len = 50 < block_q, so Q uses tailq_; KV still hits TMA once.
        run_test_case(1, 2, 2, 0, 50, 64, 1.0f32, false, false)?;
        Ok(())
    }
}
