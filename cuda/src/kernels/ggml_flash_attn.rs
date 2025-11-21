use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use num_traits::One;
use std::fmt;
use tract_core::internal::*;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{LibraryName, WARP_SIZE, get_cuda_view, launch_args};

const CUDA_CC_TURING: i32 = 750;
const CUDA_CC_AMPERE: i32 = 800;
const FATTN_KQ_STRIDE: usize = 256;

#[derive(Debug, Clone)]
pub struct GgmlFlashAttn;

impl fmt::Display for GgmlFlashAttn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, Copy)]
enum FlashAttnImpl {
    Vec,
    MmaF16,
}

/* ------------------------- small helpers ------------------------- */

#[inline]
fn cc() -> i32 {
    let p = cuda_context().properties();
    p.major * 100 + p.minor * 10
}

#[inline]
fn newer_than_lovelace() -> bool {
    let p = cuda_context().properties();
    p.major > 8 || (p.major == 8 && p.minor >= 9)
}

#[inline]
fn bytes_of<T>() -> usize {
    std::mem::size_of::<T>()
}

#[inline]
fn to_i32(shape: &[usize]) -> TVec<i32> {
    shape.iter().map(|&s| s as i32).collect()
}

#[inline]
fn strides_to_bytes_i32<T>(strides_elems: &[isize]) -> TVec<i32> {
    let el = bytes_of::<T>() as i32;
    strides_elems.iter().map(|&s| (s as i32) * el).collect()
}

#[derive(Debug, Clone)]
struct FlashAttnParams {
    imp: FlashAttnImpl,
    // tiling
    d: usize,
    ncols1: usize,
    ncols2: usize,
    nwarps: usize,
    // shared
    shm_bytes: usize,
    // scheduling
    kq_row_granularity: usize,
    // derived
    kernel: Arc<CudaFunction>,
}

impl GgmlFlashAttn {
    pub fn is_supported_dts(k_dt: DatumType, v_dt: DatumType) -> bool {
        (k_dt == v_dt) && matches!(k_dt, DatumType::F16)
    }

    pub fn name(&self) -> Cow<'_, str> {
        format!("{self}").into()
    }

    pub fn vec_kernel_name(
        &self,
        d: usize,
        ncols1: usize,
        k_dt: DatumType,
        v_dt: DatumType,
    ) -> TractResult<String> {
        ensure!(
            Self::is_supported_dts(k_dt, v_dt),
            "Unsupported dts K: {:?} V: {:?} for Cuda Flash Attention Op",
            k_dt,
            v_dt
        );
        Ok(format!(
            "flash_attn_vec_{}_{}_{}_{}",
            d,
            ncols1,
            DeviceTensor::tname(k_dt)?,
            DeviceTensor::tname(v_dt)?
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

    /* ------------------------- kernel planning ------------------------- */
    fn pick_ncols2(head_ratio: usize) -> usize {
        if head_ratio % 8 == 0 {
            8
        } else if head_ratio % 4 == 0 {
            4
        } else if head_ratio % 2 == 0 {
            2
        } else {
            1
        }
    }
    fn pick_ncols1(ncols2: usize, seq_q: usize, cc: i32) -> usize {
        if ncols2 <= 8 && seq_q <= 8 / ncols2 {
            8 / ncols2
        } else if seq_q <= 16 / ncols2 {
            16 / ncols2
        } else if cc == CUDA_CC_TURING || seq_q <= 32 / ncols2 {
            32 / ncols2
        } else {
            64 / ncols2
        }
    }

    fn get_flash_attn_vec_params(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
    ) -> TractResult<FlashAttnParams> {
        let qs = q.shape();
        let rank = qs.len();
        ensure!(matches!(qs[rank - 2], 1 | 2));
        let ncols1 = qs[rank - 2];
        let ncols2 = 1;
        let d = qs[rank - 1];

        let kernel_name = self.vec_kernel_name(d, ncols1, k.datum_type(), v.datum_type())?;
        let kernel = cuda_context().load_pipeline(LibraryName::GgmlFlashAttn, kernel_name)?;

        Ok(FlashAttnParams {
            imp: FlashAttnImpl::Vec,
            d,
            ncols1,
            ncols2,
            nwarps: 128 / WARP_SIZE,
            shm_bytes: 0,
            kq_row_granularity: d,
            kernel,
        })
    }

    fn get_flash_attn_mma_f16_params(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
    ) -> TractResult<FlashAttnParams> {
        let cc = cc();
        let qs = q.shape();
        let ks = k.shape();

        let d = qs[3];
        ensure!(d % 8 == 0, "flash-attn f16: head dimension (D) must be multiple of 8");

        let head_ratio = qs[1] / ks[1];
        let ncols2 = Self::pick_ncols2(head_ratio);
        let ncols1 = Self::pick_ncols1(ncols2, qs[2], cc);
        let ncols = ncols1 * ncols2;

        let ntiles = if ncols <= 8 { 1 } else { 2 };
        let cols_per_warp = ntiles * 8;
        ensure!(ncols % cols_per_warp == 0, "bad ncols vs cols_per_warp");

        let nbatch_fa = if d != 256 { 64 } else { 32 };
        let nwarps_max_x = ncols / cols_per_warp;
        let nwarps_max_y = nbatch_fa / 16;
        let nwarps = (nwarps_max_x * nwarps_max_y).min(4);

        // shared bytes
        let nbatch_k2 = d / 2;
        let nbatch_v2 = d / 2;
        let nbatch_combine = if cc == CUDA_CC_TURING && ncols1 <= 128 { 128 } else { 64 };
        let shm_kv_1stage = nbatch_fa * (nbatch_k2 + 4).max(nbatch_v2 + 4) * bytes_of::<f32>();
        let shm_kv_2stage = nbatch_fa * (nbatch_k2 + 4 + nbatch_v2 + 4) * bytes_of::<f32>();
        let shm_q = ncols * (d / 2 + 4) * bytes_of::<f32>();
        let shm_mask = ncols1 * (nbatch_fa / 2 + 4) * bytes_of::<f32>();
        let shm_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * bytes_of::<f32>();
        let shm_kv = if cc >= CUDA_CC_AMPERE { shm_kv_2stage } else { shm_kv_1stage };
        let shm_bytes = shm_combine.max(shm_q.max(shm_kv + shm_mask));

        let kernel_name = format!("flash_attn_ext_f16_{}_{}_{}", d, ncols, ncols2);
        let kernel = cuda_context().load_pipeline(LibraryName::GgmlFlashAttn, kernel_name)?;
        kernel.set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shm_bytes as i32,
        )?;

        Ok(FlashAttnParams {
            imp: FlashAttnImpl::MmaF16,
            d,
            ncols1,
            ncols2,
            nwarps,
            shm_bytes,
            kq_row_granularity: FATTN_KQ_STRIDE,
            kernel,
        })
    }

    fn choose_impl(
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        m: &DeviceTensor,
    ) -> TractResult<FlashAttnImpl> {
        let qh = q.shape()[1];
        let kh = k.shape()[1];
        ensure!(qh % kh == 0);
        let head_ratio = qh / kh;

        let newer = newer_than_lovelace();

        ensure!(
            matches!(k.shape()[3], 64 | 80 | 96 | 112 | 128 | 256) && k.shape()[3] == v.shape()[3],
            "No kernel for K/V D={} (must match and be one of 64|80|96|112|128|256)",
            k.shape()[3]
        );

        // Batched mask support could be done by modifying KV_max kernel
        ensure!(m.shape()[..2] == [1, 1]);

        let can_vec = q.shape()[3] % 64 == 0;
        let mut best = FlashAttnImpl::MmaF16;

        if can_vec {
            if newer
                && q.shape()[2] == 1
                && q.shape()[0] == 1
                && !(head_ratio > 4 && k.shape()[2] >= 8192)
            {
                best = FlashAttnImpl::Vec;
            }
            if (head_ratio % 2 != 0) && q.shape()[2] == 1 {
                best = FlashAttnImpl::Vec; // GQA-specific case
            }
        }
        Ok(best)
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
        ensure!(q.datum_type() == DatumType::F32 && q.datum_type() == out.datum_type());
        ensure!(out.shape() == self.output_shape(q.shape(), k.shape(), v.shape())?.as_slice());

        match Self::choose_impl(q, k, v, mask)? {
            FlashAttnImpl::Vec => {
                let params = self.get_flash_attn_vec_params(q, k, v)?;
                self.launch_with_plan(stream, q, k, v, mask, scale, out, params)
            }
            FlashAttnImpl::MmaF16 => {
                let params = self.get_flash_attn_mma_f16_params(q, k)?;
                self.launch_with_plan(stream, q, k, v, mask, scale, out, params)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_with_plan(
        &self,
        stream: &TractCudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: &DeviceTensor,
        scale: f32,
        out: &DeviceTensor,
        params: FlashAttnParams,
    ) -> TractResult<()> {
        // quick invariants shared by both variants
        ensure!(mask.shape()[2] >= q.shape()[2].next_multiple_of(16));
        ensure!(k.shape()[2] % FATTN_KQ_STRIDE == 0, "Incorrect KV cache padding");

        let qv = get_cuda_view(q);
        let kv = get_cuda_view(k);
        let vv = get_cuda_view(v);
        let mv = get_cuda_view(mask);
        let ov = get_cuda_view(out);

        // Grid/block sizing & occupancy
        let ncols = params.ncols1 * params.ncols2;
        let ntiles_x = q.shape()[2].div_ceil(params.ncols1);
        let ntiles_total = ntiles_x * (q.shape()[1] / params.ncols2) * q.shape()[0];

        // mask-to-KV-max
        let kv_max = if q.shape()[2] >= 1024 || q.shape()[0] > 1 {
            let mask_s2_div2 = mask.strides()[2] / 2;
            let mask_s0_div2 = mask.strides()[0] / 2;

            let blocks_num = (ntiles_x as _, q.shape()[0] as _, 1);
            let blocks_dim = ((FATTN_KQ_STRIDE / 2) as _, 1, 1);
            let ne_kv_max = ntiles_x * q.shape()[0];
            let iter_k = k.shape()[2] / FATTN_KQ_STRIDE;

            let kv_max = DeviceTensor::uninitialized_dt(DatumType::I64, &[ne_kv_max])?;
            let kv_max_v = get_cuda_view(&kv_max);
            let func = cuda_context().load_pipeline(
                LibraryName::GgmlFlashAttn,
                format!("flash_attn_mask_to_KV_max_{}", params.ncols1),
            )?;

            let mut la = stream.launch_builder(&func);
            la.arg(&mv);
            la.arg(&kv_max_v);
            la.arg(&iter_k);
            la.arg(&mask_s2_div2);
            la.arg(&mask_s0_div2);
            let cfg =
                LaunchConfig { grid_dim: blocks_num, block_dim: blocks_dim, shared_mem_bytes: 0 };

            unsafe {
                la.launch(cfg);
            }
            Some(kv_max)
        } else {
            None
        };

        // occupancy & parallel layout
        let props = cuda_context().properties();
        let nsm = props.multiProcessorCount as usize;
        let block_dim = (WARP_SIZE as u32, params.nwarps as u32, 1);

        let max_blocks_per_sm = params.kernel.occupancy_max_active_blocks_per_multiprocessor(
            WARP_SIZE as u32 * params.nwarps as u32,
            params.shm_bytes,
            None,
        )? as usize;

        let mut parallel_blocks = max_blocks_per_sm;

        let (blocks_num, dst_tmp, dst_tmp_meta) = if matches!(params.imp, FlashAttnImpl::MmaF16) {
            let max_blocks = max_blocks_per_sm * nsm;
            let tiles_nwaves = ntiles_total.div_ceil(max_blocks);
            let tiles_eff_pct = 100 * ntiles_total / (max_blocks * tiles_nwaves);
            let newer = newer_than_lovelace();
            let use_stream_k = newer || tiles_eff_pct < 75;

            let blocks_num = (if use_stream_k { max_blocks } else { ntiles_total } as u32, 1, 1);
            let dst_tmp_meta = DeviceTensor::uninitialized_dt(
                DatumType::F32,
                &[2 * blocks_num.0 as usize * ncols * (2 * 2 + params.d) * bytes_of::<f32>()],
            )?;
            (blocks_num, None, Some(dst_tmp_meta))
        } else {
            ensure!(k.shape()[k.rank() - 2] % params.kq_row_granularity == 0);

            let ntiles_kq = k.shape()[k.rank() - 2].div_ceil(params.kq_row_granularity);
            let pb_min = parallel_blocks.min(ntiles_kq);

            // try to improve tail efficiency
            let blocks_per_wave = nsm * max_blocks_per_sm;
            let mut nwaves_best = 0;
            let mut eff_best = 0;
            for pb in (pb_min..=ntiles_kq) {
                let nblocks_total = ntiles_total * pb;
                let nwaves = nblocks_total.div_ceil(blocks_per_wave);
                let eff = 100 * nblocks_total / (nwaves * blocks_per_wave);
                if eff_best >= 95 && nwaves > nwaves_best {
                    break;
                }
                if eff > eff_best {
                    nwaves_best = nwaves;
                    eff_best = eff;
                    parallel_blocks = pb;
                }
            }

            ensure!(
                parallel_blocks > 1,
                "Unsupported config: Output won't be untransposed if we don't enter vec fixup kernel"
            );
            let blocks_num = (
                ntiles_x as u32,
                parallel_blocks as u32,
                (q.shape()[1] / params.ncols2 * q.shape()[0]) as u32,
            );

            (
                blocks_num,
                Some(DeviceTensor::uninitialized_dt(
                    DatumType::F32,
                    &[parallel_blocks * out.shape().iter().product::<usize>()],
                )?),
                Some(DeviceTensor::uninitialized_dt(
                    DatumType::F32,
                    &[2 * parallel_blocks * out.shape()[..3].iter().product::<usize>()],
                )?),
            )
        };

        ensure!(block_dim.0 % WARP_SIZE as u32 == 0);

        // Shapes/strides for kernel
        let q_shape_i32 = to_i32(q.shape());
        let k_shape_i32 = to_i32(k.shape());
        let mask_shape_i32 = to_i32(mask.shape());
        let q_strides_b = strides_to_bytes_i32::<f32>(q.strides());
        let k_strides_b = strides_to_bytes_i32::<f16>(k.strides());
        let v_strides_b = strides_to_bytes_i32::<f16>(v.strides());
        let mask_strides_b = strides_to_bytes_i32::<f16>(mask.strides());

        let null_ptr = stream.null::<u8>()?;
        let kv_max_v = kv_max.as_ref().map(get_cuda_view).unwrap_or_else(|| null_ptr.as_view());
        let dst_tmp_v = dst_tmp.as_ref().map(get_cuda_view).unwrap_or_else(|| null_ptr.as_view());
        let dst_tmp_meta_v =
            dst_tmp_meta.as_ref().map(get_cuda_view).unwrap_or_else(|| null_ptr.as_view());

        // main kernel
        let mut la = stream.launch_builder(&params.kernel);
        la.arg(&qv);
        la.arg(&kv);
        la.arg(&vv);
        la.arg(&mv);
        la.arg(&kv_max_v);
        la.arg(if matches!(params.imp, FlashAttnImpl::Vec) { &dst_tmp_v } else { &ov });
        la.arg(&dst_tmp_meta_v);
        la.arg(&scale);
        la.set_slice(&q_shape_i32);
        la.set_slice(&q_strides_b[..3]);
        la.set_slice(&k_shape_i32);
        la.set_slice(&k_strides_b[..3]);
        la.set_slice(&v_strides_b[..3]);
        la.set_slice(&mask_shape_i32[..3]);
        la.set_slice(&mask_strides_b[..3]);

        let cfg = LaunchConfig {
            grid_dim: blocks_num,
            block_dim,
            shared_mem_bytes: params.shm_bytes as u32,
        };

        unsafe {
            la.launch(cfg);
        }

        // fixups
        if matches!(params.imp, FlashAttnImpl::MmaF16) {
            if ntiles_total as u32 % cfg.grid_dim.0 != 0 {
                let f = cuda_context().load_pipeline(
                    LibraryName::GgmlFlashAttn,
                    format!("flash_attn_stream_k_fixup_{}_{}_{}", params.d, ncols, params.ncols2),
                )?;
                let mut la = stream.launch_builder(&f);
                la.arg(&ov);
                la.arg(&dst_tmp_meta_v);
                la.set_slice(&q_shape_i32[..3]);
                la.arg(&k_shape_i32[2]);
                let cfg = LaunchConfig {
                    grid_dim: (cfg.grid_dim.0, params.ncols1 as _, params.ncols2 as _),
                    block_dim: (params.d as _, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    la.launch(cfg);
                }
            }
        } else {
            let f = cuda_context().load_pipeline(
                LibraryName::GgmlFlashAttn,
                format!("flash_attn_combine_results_{}", params.d),
            )?;
            let mut la = stream.launch_builder(&f);
            la.arg(&dst_tmp_v);
            la.arg(&dst_tmp_meta_v);
            la.arg(&ov);
            la.arg(&parallel_blocks);
            let cfg = LaunchConfig {
                grid_dim: (q_shape_i32[2] as _, q_shape_i32[1] as _, q_shape_i32[0] as _),
                block_dim: (params.d as _, 1, 1),
                shared_mem_bytes: (parallel_blocks * 2 * bytes_of::<f32>()) as u32,
            };
            unsafe {
                la.launch(cfg);
            }
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

    fn pad_f16_tensor(
        a: &Tensor,
        axis: usize,
        block_size: usize,
        value: f16,
    ) -> TractResult<Tensor> {
        let mut shape = a.shape().to_owned();
        let old_value = shape[axis];
        shape[axis] = shape[axis].next_multiple_of(block_size);
        let mut padded_a = Tensor::zero::<f16>(&shape)?;
        padded_a.fill_t(value);
        padded_a
            .to_array_view_mut::<f16>()?
            .slice_axis_move(tract_ndarray::Axis(axis), (0..old_value).into())
            .assign(&a.to_array_view::<f16>()?);
        Ok(padded_a)
    }
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
                &(0..q_len).map(|f| f as f32 / q_len as f32).collect::<Vec<_>>(),
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

            let cuda_output = GgmlFlashAttn.eval(
                stream,
                &q.clone().into_device()?,
                &pad_f16_tensor(&k, 2, FATTN_KQ_STRIDE, f16::from_f32(0f32))?.into_device()?,
                &pad_f16_tensor(&v, 2, FATTN_KQ_STRIDE, f16::from_f32(0f32))?.into_device()?,
                &&pad_f16_tensor(
                    &pad_f16_tensor(&m, 3, FATTN_KQ_STRIDE, -f16::infinity())?,
                    2,
                    16,
                    -f16::infinity(),
                )?
                .into_device()?,
                scale,
            )?;

            let ref_output = Sdpa {
                scale: Some(scale.into()),
                datum_type: DatumType::F32,
                acc_datum_type: DatumType::F32,
                is_causal: false,
            }
            .eval(tvec!(q.into(), k.into(), v.into(), m.into()))?;

            cuda_output.to_host()?.close_enough(&ref_output[0], Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_fattn_vec() -> TractResult<()> {
        run_test_case(1, 1, 1, 0, 1, 64, 1.0f32)?;
        run_test_case(1, 32, 8, 0, 1, 64, 1.0f32)?;
        run_test_case(1, 8, 8, 0, 1, 128, 1.0f32)?;
        run_test_case(2, 8, 8, 0, 1, 64, 1.0f32)?;
        run_test_case(1, 1, 1, 3, 1, 128, 1.0f32)?;
        run_test_case(2, 24, 8, 3, 1, 64, 1.0f32)?;
        Ok(())
    }

    #[test]
    fn test_fattn_mma_f16() -> TractResult<()> {
        run_test_case(1, 1, 1, 0, 1, 64, 1.0f32)?;
        run_test_case(1, 8, 8, 1, 1, 64, 1.0f32)?;
        run_test_case(2, 4, 2, 1, 1, 128, 1.0f32)?;
        run_test_case(2, 8, 8, 0, 1, 128, 1.0f32)?;
        run_test_case(1, 1, 1, 3, 2, 64, 1.0f32)?;
        run_test_case(2, 2, 1, 3, 1, 128, 1.0f32)?;
        Ok(())
    }
}
