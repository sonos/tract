use cudarc::driver::result::occupancy::max_active_block_per_multiprocessor;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::runtime::sys::cudaOccupancyMaxActiveBlocksPerMultiprocessor;
use num_traits::One;
use std::fmt;
use tract_core::internal::*;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{LibraryName, WARP_SIZE, get_cuda_view, launch_args};

static N_THREADS: u32 = 128;
static N_WARPS: u32 = 4;

#[derive(Debug, Clone)]
pub struct GgmlFlashAttnOp {
    scale: f32,
    is_causal: bool,
}

impl fmt::Display for GgmlFlashAttnOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

enum FlashAttnImpl {
    Vec,
    MmaF16,
}

impl GgmlFlashAttnOp {
    pub fn is_supported_dts(k_dt: DatumType, v_dt: DatumType) -> bool {
        (k_dt == v_dt) && matches!(k_dt, DatumType::F16 | DatumType::Opaque)
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
            "flash_attn_vec_{}_{}_{:?}_{:?}",
            d,
            ncols1,
            DeviceTensor::tname(k_dt),
            DeviceTensor::tname(v_dt)
        ))
    }

    pub fn output_shape<D: DimLike + One + Copy>(
        &self,
        q: &[D],
        k: &[D],
        v: &[D],
    ) -> TractResult<TVec<D>> {
        ensure!(q.len() == 3 || q.len() == 4, "Q rank must be 3 or 4 (got {})", q.len());
        ensure!(
            k.len() == q.len() && v.len() == q.len(),
            "K and V must have the same rank as Q (Q={}, K={}, V={})",
            q.len(),
            k.len(),
            v.len()
        );

        match (q, k, v) {
            // Rank-3: [B, S, D]
            ([b, seq_len, _], [_, _, _], [_, _, out_dim]) => {
                Ok(tvec![*b, 1.into(), *seq_len, *out_dim])
            }

            // Rank-4: [B, H, S, D]
            ([b, q_heads, seq_len, _], [_, k_heads, _, _], [_, v_heads, _, out_dim]) => {
                let qh = (*q_heads).to_i64()?;
                let kh = (*k_heads).to_i64()?;
                let vh = (*v_heads).to_i64()?;

                ensure!(
                    kh == vh,
                    "K and V must have the same number of heads (K={}, V={})",
                    kh,
                    vh
                );
                ensure!(kh > 0, "K/V heads must be > 0 (got {})", kh);
                ensure!(qh % kh == 0, "Q heads ({}) must be a multiple of K/V heads ({})", qh, kh);

                Ok(tvec![*b, *q_heads, *seq_len, *out_dim])
            }

            _ => bail!("Inconsistent shapes: expected [B,S,D] or [B,H,S,D] for Q/K/V."),
        }
    }

    fn find_fattn_kernel(
        q_shape: &[usize],
        k_shape: &[usize],
        v_shape: &[usize],
        m_shape: Option<&[usize]>,
        k_dt: DatumType,
        v_dt: DatumType,
    ) -> TractResult<FlashAttnImpl> {
        ensure!(q_shape[1] % k_shape[1] == 0);
        let head_ratio = q_shape[1] / k_shape[1];

        let ctxt = cuda_context();
        let prop = ctxt.properties();
        let newer_than_lovelace = prop.major > 8 || (prop.major == 8 && prop.minor > 9);

        match k_shape[3] {
            64 | 80 | 96 | 112 | 128 | 256 => ensure!(k_shape[3] == v_shape[3]),
            576 => ensure!((v_shape[3] == 512) && (head_ratio % 16 == 0)),
            _ => bail!("No kernel available for k_shape[3] = {}", k_shape[3]),
        }

        ensure!(m_shape.is_none_or(|shape| shape[1] != 1));

        let can_use_vector_kernel = q_shape[3] <= 256 && q_shape[3] % 64 == 0;
        let mut best = FlashAttnImpl::MmaF16;

        if can_use_vector_kernel {
            if (k_dt == DatumType::F16 && v_dt == DatumType::F16) {
                if (newer_than_lovelace
                    && q_shape[2] == 1
                    && q_shape[0] == 1
                    && !(head_ratio > 4 && k_shape[2] >= 8192))
                {
                    best = FlashAttnImpl::Vec;
                }
            } else {
                if (newer_than_lovelace) {
                    if (q_shape[2] <= 2) {
                        best = FlashAttnImpl::Vec;
                    }
                } else {
                    if (q_shape[2] == 1) {
                        best = FlashAttnImpl::Vec;
                    }
                }
            }
            if ((head_ratio % 2 != 0 || m_shape.is_none()) && q_shape[2] == 1) {
                best = FlashAttnImpl::Vec; // GQA-specific optimizations in the mma kernel do not apply.
            }
        }
        Ok(best)
    }

    pub fn launch_generic_flash_attn(
        &self,
        stream: &TractCudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: Option<&DeviceTensor>,
        out: &DeviceTensor,
        func: Arc<CudaFunction>,
        d: usize,
        kq_row_granularity: usize,
        ncols1: usize,
        ncols2: usize,
        nbytes_shared: usize,
        stream_k: bool,
    ) -> TractResult<()> {
        let ncols = ncols1 * ncols2;
        let is_mla = d == 512;

        ensure!(mask.is_none_or(|m| {
            let n = m.shape()[m.rank() - 2];
            n >= n.div_ceil(16) * 16
        }));
        ensure!(k.shape()[k.rank() - 2] % 256 == 0, "Incorrect KV cache padding");

        let q_rank = q.rank();
        let q_shape = q.shape();
        let ntiles_x = q_shape[q_rank - 2].div_ceil(ncols1);
        let ntiles_total = ntiles_x
            * (q_shape[q_rank - 3] / ncols2)
            * if q_rank == 4 { q_shape[q_rank - 4] } else { 1 };

        // TODO: implement mask optim
        let ctxt = cuda_context();
        let prop = ctxt.properties();
        let nsm = prop.multiProcessorCount as usize;
        let block_dim: (u32, u32, u32) = (WARP_SIZE as _, N_WARPS as _, 1);

        let max_blocks_per_sm = func.occupancy_max_active_blocks_per_multiprocessor(
            WARP_SIZE as u32 * N_WARPS,
            nbytes_shared,
            None,
        )? as usize;
        let mut parallel_blocks = max_blocks_per_sm;

        let (blocks_num, dst_tmp, dst_tmp_meta) = if stream_k {
            // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
            let max_blocks = max_blocks_per_sm * nsm;

            let tiles_nwaves = ntiles_total.div_ceil(max_blocks);
            let tiles_efficiency_percent = 100 * ntiles_total / (max_blocks * tiles_nwaves);

            let newer_than_lovelace = prop.major > 8 || (prop.major == 8 && prop.minor > 9);
            let use_stream_k = newer_than_lovelace || tiles_efficiency_percent < 75;

            let blocks_num = (if use_stream_k { max_blocks } else { ntiles_total } as u32, 1, 1);
            let dst_tmp_meta = DeviceTensor::uninitialized_dt(
                DatumType::U8,
                &[blocks_num.0 as usize * ncols * (2 * 2 + d) * size_of::<f32>()],
            )?;

            (blocks_num, None, Some(dst_tmp_meta))
        } else {
            ensure!(k.shape()[k.rank() - 2] % kq_row_granularity == 0);
            let ntiles_kq = k.shape()[k.rank() - 2] / kq_row_granularity; // Max. number of parallel blocks limited by tensor size.

            // parallel_blocks must not be larger than what the tensor size allows:
            parallel_blocks = parallel_blocks.min(ntiles_kq);

            // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
            // Test whether parallel_blocks can be set to a higher value for better efficiency.
            let blocks_per_wave = nsm * max_blocks_per_sm;
            let mut nwaves_best = 0;
            let mut efficiency_percent_best = 0;
            for parallel_blocks_test in (parallel_blocks..=ntiles_kq).step_by(parallel_blocks) {
                let nblocks_total = ntiles_total * parallel_blocks_test;
                let nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
                let efficiency_percent = 100 * nblocks_total / (nwaves * blocks_per_wave);

                // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
                if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                    break;
                }

                if (efficiency_percent > efficiency_percent_best) {
                    nwaves_best = nwaves;
                    efficiency_percent_best = efficiency_percent;
                    parallel_blocks = parallel_blocks_test;
                }
            }

            let blocks_num = (
                ntiles_x as u32,
                parallel_blocks as u32,
                (q_shape[q_rank - 3] * if q_rank == 4 { q_shape[q_rank - 4] } else { 1 }) as u32,
            );
            let (dst_tmp, dst_tmp_meta) = if parallel_blocks > 1 {
                (
                    Some(DeviceTensor::uninitialized_dt(
                        DatumType::U8,
                        &[parallel_blocks * out.shape().iter().product::<usize>()],
                    )?),
                    Some(DeviceTensor::uninitialized_dt(
                        DatumType::U8,
                        &[parallel_blocks * out.shape()[..3].iter().product::<usize>()],
                    )?),
                )
            } else {
                (None, None)
            };
            (blocks_num, dst_tmp, dst_tmp_meta)
        };

        ensure!(block_dim.0 % WARP_SIZE as u32 == 0);

        let q_shape = q.shape().iter().map(|s| *s as i32).collect_vec();
        let k_shape = k.shape().iter().map(|s| *s as i32).collect_vec();
        let mask_shape = mask
            .map(|m| m.shape().iter().map(|s| *s as i32).collect_vec())
            .unwrap_or_else(|| vec![0, 0, 0, 0]);

        let q_strides =
            q.strides().iter().map(|s| *s as i32 * size_of::<f32>() as i32).collect_vec();
        let k_strides =
            k.strides().iter().map(|s| *s as i32 * size_of::<f32>() as i32).collect_vec();
        let v_strides =
            v.strides().iter().map(|s| *s as i32 * size_of::<f32>() as i32).collect_vec();
        let v_strides =
            v.strides().iter().map(|s| *s as i32 * size_of::<f32>() as i32).collect_vec();
        let mask_strides = mask
            .map(|m| m.strides().iter().map(|s| *s as i32 * size_of::<f32>() as i32).collect_vec())
            .unwrap_or_else(|| vec![0, 0, 0, 0]);

        let null_ptr = stream.null::<u8>()?;

        let q_view = get_cuda_view(q);
        let k_view = get_cuda_view(k);
        let v_view = get_cuda_view(v);
        let mask_view = mask.map(|t| get_cuda_view(t)).unwrap_or_else(|| null_ptr.as_view());
        let out_view = get_cuda_view(out);
        let dst_tmp_view =
            dst_tmp.as_ref().map(|t| get_cuda_view(t)).unwrap_or_else(|| null_ptr.as_view());
        let dst_tmp_meta_view =
            dst_tmp_meta.as_ref().map(|t| get_cuda_view(t)).unwrap_or_else(|| null_ptr.as_view());

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&q_view);
        launch_args.arg(&k_view);
        launch_args.arg(&mask_view);
        launch_args.arg(&null_ptr); // KV_max
        launch_args.arg(if !stream_k && parallel_blocks > 1 { &dst_tmp_view } else { &out_view });
        launch_args.arg(&self.scale);
        launch_args.set_slice(&q_shape);
        launch_args.set_slice(&q_strides[..3]);
        launch_args.set_slice(&k_shape);
        launch_args.set_slice(&k_strides[..3]);
        launch_args.set_slice(&v_strides[..3]);
        launch_args.set_slice(&mask_shape[..3]);
        launch_args.set_slice(&mask_strides[..3]);

        let cfg =
            LaunchConfig { grid_dim: blocks_num, block_dim, shared_mem_bytes: nbytes_shared as _ };

        unsafe {
            launch_args.launch(cfg);
        }

        if stream_k {
            if ntiles_total as u32 % blocks_num.0 != 0 {
                let func = cuda_context().load_pipeline(
                    LibraryName::FlashAttn,
                    format!("flash_attn_stream_k_fixup_{}_{}_{}", d, ncols1, ncols2),
                )?;

                let mut launch_args = stream.launch_builder(&func);
                launch_args.arg(&out_view);
                launch_args.arg(&dst_tmp_meta_view);
                launch_args.set_slice(&q_shape[..3]);
                launch_args.arg(&k_shape[2]);

                let cfg = LaunchConfig {
                    grid_dim: (d as _, 1, 1),
                    block_dim: (blocks_num.0, ncols1 as _, ncols2 as _),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    launch_args.launch(cfg);
                }
            }
        } else if parallel_blocks > 1 {
            let func = cuda_context().load_pipeline(
                LibraryName::FlashAttn,
                format!("flash_attn_combine_results_{}", d),
            )?;

            let mut launch_args = stream.launch_builder(&func);
            launch_args.arg(&dst_tmp_view);
            launch_args.arg(&dst_tmp_meta_view);
            launch_args.arg(&out_view);
            launch_args.arg(&parallel_blocks);

            let cfg = LaunchConfig {
                grid_dim: (d as _, 1, 1),
                block_dim: (q_shape[2] as _, q_shape[1] as _, q_shape[0] as _),
                shared_mem_bytes: (parallel_blocks * 2 * size_of::<f32>()) as u32,
            };
            unsafe {
                launch_args.launch(cfg);
            }
        }

        Ok(())
    }

    pub fn launch_flash_attn_vec(
        &self,
        stream: &TractCudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: Option<&DeviceTensor>,
        out: &DeviceTensor,
    ) -> TractResult<()> {
        let q_shape = q.shape();
        let rank = q_shape.len();
        ensure!(matches!(q_shape[rank - 2], 1 | 2));

        let ncols1 = q_shape[rank - 2];
        let ncols2 = 1;
        let d = q_shape[rank - 1];
        let kernel_name = self.vec_kernel_name(d, ncols1, k.datum_type(), v.datum_type())?;
        let func = cuda_context().load_pipeline(LibraryName::FlashAttn, kernel_name)?;

        self.launch_generic_flash_attn(
            stream, q, k, v, mask, out, func, d, d, ncols1, ncols2, 0, false,
        )
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: Option<&DeviceTensor>,
    ) -> TractResult<DeviceTensor> {
        ensure!(!self.is_causal || mask.is_none());
        let output = unsafe {
            DeviceTensor::uninitialized_dt(
                q.datum_type(),
                &self.output_shape(q.shape(), k.shape(), v.shape())?,
            )?
        };
        self.dispatch_eval(stream, q, k, v, mask, &output)?;
        stream.synchronize()?;

        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: Option<&DeviceTensor>,
        out: &DeviceTensor,
    ) -> TractResult<()> {
        if q.rank() == 3 {
            let q_shape = q.shape();
            let k_shape = k.shape();
            let v_shape = v.shape();
            let out_shape = out.shape();
            q.reshaped(tvec!(1, q_shape[0], q_shape[1], q_shape[2]))?;
            k.reshaped(tvec!(1, k_shape[0], k_shape[1], k_shape[2]))?;
            v.reshaped(tvec!(1, v_shape[0], v_shape[1], v_shape[2]))?;
            out.reshaped(tvec!(1, out_shape[0], out_shape[1], out_shape[2]))?;
        }

        if let Some(m) = mask {
            let shape = m.shape();
            m.reshaped(tvec!(1, shape[0], shape[1], shape[2]))?;
        }
        let kernel = Self::find_fattn_kernel(
            q.shape(),
            k.shape(),
            v.shape(),
            mask.map(|m| m.shape()),
            k.datum_type(),
            v.datum_type(),
        )?;
        match kernel {
            FlashAttnImpl::Vec => self.launch_flash_attn_vec(stream, q, k, v, mask, out),
            FlashAttnImpl::MmaF16 => todo!(),
        }
    }
}
