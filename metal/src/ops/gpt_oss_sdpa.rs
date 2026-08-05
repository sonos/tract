//! Metal lowering of the fused GPT-OSS attention (`GptOssSdpa`).
//!
//! The op-state owns device-resident K/V capacity buffers ([1, Hkv, cap, D]
//! f16, geometric growth). Each step appends only the new rows (49 KB/token
//! for gpt-oss-20b instead of the O(T) concat copy) and emits the caches as
//! zero-copy strided views of the capacity buffer: appends only ever write
//! past `len`, so views held by the caller stay valid; growth reallocates and
//! the old buffer stays alive through the views' Arc.
//!
//! Attention runs per kv-head on contiguous sub-views (each head's region of
//! the capacity buffer is contiguous), so the standard GGML gemm kernels
//! apply unchanged: QK gemm, the fused scale+mask+sinks softmax kernel, AV
//! gemm.

use std::sync::Arc;

use anyhow::ensure;
use tract_core::internal::*;
use tract_gpu::device::get_context;
use tract_gpu::tensor::{DeviceArenaView, DeviceTensor, DeviceTensorExt, OwnedDeviceTensor};
use tract_gpu::utils::facts_to_device_facts;
use tract_transformers::ops::gpt_oss_sdpa::GptOssSdpa;

use crate::kernels::matmul::{GemmDispatchParams, GemmKernel, GgmlGemm};
use crate::kernels::moe::{
    GptOssFlashAttnDims, dispatch_gpt_oss_flash_attn_f16, dispatch_gpt_oss_sinks_softmax_f16,
    flash_attn_scratch_len,
};
use crate::utils::get_metal_buffer;

const SEQ_AXIS: usize = 2;
/// Step sizes up to this may run the fused flash-attention kernel; larger
/// (prefill-sized) steps always use the batched-gemm pipeline.
const FLASH_MAX_S: usize = 8;
/// Context length where flash decode takes over from the batched gemms.
/// Benchmarked 2026-08-05 on gpt-oss-20b (M-series): the batched
/// GGML mm/gemv pipeline beat the flash kernel at every length tried (74,
/// 2800, 5.6k, 11k ctx: e.g. 24.3 vs 21.1 tok/s at 11k), so flash is
/// disabled by default and kept for future tuning. Enable with
/// TRACT_METAL_GPT_OSS_FLASH_MIN_T=<t>.
const FLASH_MIN_T: usize = usize::MAX;

fn flash_min_t() -> usize {
    std::env::var("TRACT_METAL_GPT_OSS_FLASH_MIN_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(FLASH_MIN_T)
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MetalGptOssSdpa {
    pub scale_bits: u32,
}

impl Op for MetalGptOssSdpa {
    fn name(&self) -> StaticName {
        "MetalGptOssSdpa".into()
    }
    op_as_typed_op!();
}

impl TypedOp for MetalGptOssSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |facts| {
            let (q, k_new, _v_new, k_cache, v_cache) =
                (facts[0], facts[1], facts[2], facts[3], facts[4]);
            let total = k_cache.shape[SEQ_AXIS].clone() + k_new.shape[SEQ_AXIS].clone();
            let mut k_out = k_cache.without_value();
            k_out.shape.set(SEQ_AXIS, total.clone());
            let mut v_out = v_cache.without_value();
            v_out.shape.set(SEQ_AXIS, total);
            Ok(tvec!(q.without_value(), k_out, v_out))
        })
    }
    as_op!();
}

impl EvalOp for MetalGptOssSdpa {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(MetalGptOssSdpaState {
            scale: f32::from_bits(self.scale_bits),
            k: DeviceKvBuffer::default(),
            // V is stored transposed: the batched GGML AV gemm needs its k
            // axis (the sequence) contiguous, and the flash decode kernel
            // reads V^T rows contiguously along each key block.
            v: DeviceKvBuffer::transposed(),
            flash_scratch: None,
        })))
    }
}

/// One device-side capacity buffer holding a logical [1, Hkv, len, D] cache.
///
/// Physical layout is either seq-major ([1, Hkv, cap, D], K) or transposed
/// ([1, Hkv, D, cap], V): the GGML gemm kernels want B as [n, k] with k
/// contiguous, and for AV that k is the sequence axis. All views present the
/// logical [1, Hkv, len, D] shape; only strides differ.
#[derive(Clone, Default)]
struct DeviceKvBuffer {
    buf: Option<Arc<Box<dyn OwnedDeviceTensor>>>,
    hkv: usize,
    d: usize,
    cap: usize,
    len: usize,
    transposed: bool,
}

impl std::fmt::Debug for DeviceKvBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "DeviceKvBuffer(len={}, cap={}, transposed={})",
            self.len, self.cap, self.transposed
        )
    }
}

impl DeviceKvBuffer {
    fn transposed() -> Self {
        Self { transposed: true, ..Self::default() }
    }

    /// Strides presenting the physical buffer (capacity `cap`) as the logical
    /// [1, Hkv, seq, D] axis order.
    fn logical_strides(&self, cap: usize) -> TVec<isize> {
        if self.transposed {
            tvec![
                (self.hkv * self.d * cap) as isize,
                (self.d * cap) as isize,
                1,
                cap as isize
            ]
        } else {
            natural_strides(&[1, self.hkv, cap, self.d])
        }
    }

    /// Element stride between consecutive sequence positions.
    fn seq_elem_stride(&self) -> usize {
        if self.transposed { 1 } else { self.d }
    }

    fn full_view(&self) -> TractResult<DeviceTensor> {
        let buf = self.buf.as_ref().context("empty kv buffer")?;
        Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
            buf.clone(),
            f16::datum_type(),
            tvec![1, self.hkv, self.cap, self.d],
            self.logical_strides(self.cap),
            0,
        )?))
    }

    /// Valid region as a strided view [1, Hkv, len, D].
    fn valid_view(&self) -> TractResult<DeviceTensor> {
        let buf = self.buf.as_ref().context("empty kv buffer")?;
        Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
            buf.clone(),
            f16::datum_type(),
            tvec![1, self.hkv, self.len, self.d],
            self.logical_strides(self.cap),
            0,
        )?))
    }

    /// The valid region as the B operand of a batched GGML gemm
    /// (transpose_b, B = [batch, n, k] with k contiguous): [Hkv, len, D] for
    /// the seq-major layout, [Hkv, D, len] for the transposed one.
    fn gemm_b_view(&self) -> TractResult<DeviceTensor> {
        let buf = self.buf.as_ref().context("empty kv buffer")?;
        let (shape, strides): (TVec<usize>, TVec<isize>) = if self.transposed {
            (
                tvec![self.hkv, self.d, self.len],
                tvec![(self.d * self.cap) as isize, self.cap as isize, 1],
            )
        } else {
            (
                tvec![self.hkv, self.len, self.d],
                tvec![(self.cap * self.d) as isize, self.d as isize, 1],
            )
        };
        Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
            buf.clone(),
            f16::datum_type(),
            shape,
            strides,
            0,
        )?))
    }

    /// Contiguous [T, D] view of one head's valid region (seq-major only).
    fn head_view(&self, h: usize) -> TractResult<DeviceTensor> {
        ensure!(!self.transposed, "head_view is seq-major only");
        let buf = self.buf.as_ref().context("empty kv buffer")?;
        Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
            buf.clone(),
            f16::datum_type(),
            tvec![1, self.len, self.d],
            natural_strides(&[1, self.len, self.d]),
            h * self.cap * self.d * f16::datum_type().size_of(),
        )?))
    }

    fn alloc(&self, hkv: usize, d: usize, cap: usize) -> TractResult<Arc<Box<dyn OwnedDeviceTensor>>> {
        let shape: [usize; 4] =
            if self.transposed { [1, hkv, d, cap] } else { [1, hkv, cap, d] };
        Ok(Arc::new(get_context()?.uninitialized_device_tensor(&shape, f16::datum_type())?))
    }

    /// Append `chunk` ([1, Hkv, new, D] f16 device tensor) past `len`.
    fn append(&mut self, chunk: &DeviceTensor) -> TractResult<()> {
        let shape = chunk.shape();
        ensure!(shape.len() == 4 && shape[0] == 1);
        let (hkv, new, d) = (shape[1], shape[2], shape[3]);
        let ctx = get_context()?;
        if self.buf.is_none() {
            let cap = (new * 2).max(256);
            self.hkv = hkv;
            self.d = d;
            self.buf = Some(self.alloc(hkv, d, cap)?);
            self.cap = cap;
            self.len = 0;
        }
        ensure!(hkv == self.hkv && d == self.d, "kv geometry changed");
        // The copy_nd kernels require the output innermost axis contiguous,
        // so copies into the transposed layout are expressed in its physical
        // axis order [1, Hkv, D, seq] (the input side is fully strided).
        let permuted = |s: &[isize]| -> TVec<isize> { tvec![s[0], s[1], s[3], s[2]] };
        if self.len + new > self.cap {
            let new_cap = (self.cap * 2).max(self.len + new);
            let grown = self.alloc(hkv, d, new_cap)?;
            let old_valid = self.valid_view()?;
            let grown_strides = self.logical_strides(new_cap);
            let grown_view = DeviceTensor::ArenaView(DeviceArenaView::from_owned(
                grown.clone(),
                f16::datum_type(),
                tvec![1, hkv, self.len, d],
                grown_strides.clone(),
                0,
            )?);
            let (shape, src_strides, dst_strides) = if self.transposed {
                (
                    tvec![1, hkv, d, self.len],
                    permuted(old_valid.strides()),
                    permuted(&grown_strides),
                )
            } else {
                (tvec![1, hkv, self.len, d], old_valid.strides().into(), grown_strides)
            };
            ctx.copy_nd(&old_valid, 0, &src_strides, &grown_view, 0, &shape, &dst_strides)?;
            self.buf = Some(grown);
            self.cap = new_cap;
        }
        let dst = self.full_view()?;
        let dst_offset = self.len * self.seq_elem_stride() * f16::datum_type().size_of();
        let (shape, src_strides, dst_strides) = if self.transposed {
            (tvec![1, hkv, d, new], permuted(chunk.strides()), permuted(dst.strides()))
        } else {
            (tvec![1, hkv, new, d], chunk.strides().into(), dst.strides().into())
        };
        ctx.copy_nd(chunk, 0, &src_strides, &dst, dst_offset, &shape, &dst_strides)?;
        self.len += new;
        Ok(())
    }

    fn reset(&mut self) {
        *self = if self.transposed { Self::transposed() } else { Self::default() };
    }
}

#[derive(Clone, Debug)]
pub struct MetalGptOssSdpaState {
    scale: f32,
    k: DeviceKvBuffer,
    v: DeviceKvBuffer,
    /// Reused f32 scratch for the flash-attention partials: allocating it
    /// per step (24 layers x ~1.5 MB per token) thrashes the Metal
    /// allocator. Grown geometrically, never shrunk.
    flash_scratch: Option<DeviceTensor>,
}

impl OpState for MetalGptOssSdpaState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 7);
        let q = inputs[0].to_device_tensor()?;
        let k_new = inputs[1].to_device_tensor()?;
        let v_new = inputs[2].to_device_tensor()?;
        let k_cache = inputs[3].to_device_tensor()?;
        let v_cache = inputs[4].to_device_tensor()?;
        let mask = inputs[5].to_device_tensor()?;
        let sinks = inputs[6].to_device_tensor()?;

        ensure!(q.datum_type() == f16::datum_type(), "q must be f16");
        let (b, hq, s_len, d) =
            (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        ensure!(b == 1, "batch 1 only");
        let hkv = k_new.shape()[1];
        let group = hq / hkv;
        let past = k_cache.shape()[SEQ_AXIS];

        // Boundary between upstream producers (rope, q-concat, mask build)
        // and this op's reads (appends, q_scratch copy): same intra-buffer
        // write->read visibility defect as everywhere else on this runtime.
        crate::with_metal_stream(|stream| stream.commit_current())?;

        // Continuation vs rebuild (fresh session / truncation / retry).
        if past != self.k.len {
            if std::env::var_os("TRACT_DEBUG_GPT_OSS_REBUILD").is_some() {
                eprintln!("gptoss-rebuild: past={past} k.len={} s_len={s_len}", self.k.len);
            }
            self.k.reset();
            self.v.reset();
            if past > 0 {
                self.k.append(k_cache)?;
                self.v.append(v_cache)?;
            }
        }
        self.k.append(k_new)?;
        self.v.append(v_new)?;
        let t_len = self.k.len;

        // Mask [.., S, T] with broadcast leading dims -> [S, T] f32.
        ensure!(mask.datum_type() == f32::datum_type(), "mask must be f32");
        let mask_shape = mask.shape();
        let mrank = mask_shape.len();
        ensure!(mask_shape[mrank - 1] == t_len, "mask keys != cache len");
        ensure!(mask_shape[..mrank - 2].iter().all(|&x| x == 1), "mask leading dims must be 1");
        let mask_2d = mask.reshaped(tvec![mask_shape[mrank - 2], t_len])?;

        // Sinks -> [Hq] f32.
        ensure!(sinks.len() == hq);
        let sinks_flat = sinks.reshaped(tvec![hq])?;

        // Output scratch (scores/probs exist only on the gemm path).
        let out = Arc::new(get_context()?.uninitialized_device_tensor(
            &[hkv, group * s_len, d],
            f16::datum_type(),
        )?);

        // A operand for QK: q's dense [1, Hq, S, D] layout is exactly the
        // batched [Hkv, group*S, D] the gemm wants, so it is used in place; a
        // non-dense q (never seen in practice) is copied to scratch first.
        let q_dense = match q {
            DeviceTensor::Owned(_) => true,
            DeviceTensor::ArenaView(view) => view.is_dense(),
        };
        let (q_a, q_a_offset) = if q_dense {
            (q.clone(), q.buffer_offset::<usize>())
        } else {
            let scratch = Arc::new(get_context()?.uninitialized_device_tensor(
                &[hkv, group * s_len, d],
                f16::datum_type(),
            )?);
            let dst = subview_all(&scratch, hkv * group * s_len, d)?;
            get_context()?.copy_nd(
                q,
                0,
                q.strides(),
                &dst,
                0,
                &tvec![1, hq, s_len, d],
                &natural_strides(&[1, hq, s_len, d]),
            )?;
            (dst, 0)
        };

        // Decode steps run the fused flash-attention kernel (one dispatch,
        // online f32 softmax, no materialized scores); prefill-sized steps
        // keep the batched-gemm pipeline, whose tiled mm kernels win when S
        // is large.
        let use_flash = s_len <= FLASH_MAX_S
            && t_len >= flash_min_t()
            && d <= 64
            && group <= 8
            && std::env::var_os("TRACT_METAL_DISABLE_GPT_OSS_FLASH").is_none();
        let f16dt = f16::datum_type();
        let gemm = GgmlGemm;
        crate::with_metal_stream(|stream| {
            // Command-buffer boundary between the cache appends (copy_nd
            // above) and the attention reads of the same buffer: the Metal
            // runtime's intra-buffer write->read visibility defect corrupts
            // this exact pattern past ~1024 tokens (same medicine as the MoE
            // routed-matmul splits).
            stream.commit_current()?;
            let k_b = self.k.gemm_b_view()?;
            let v_b = self.v.gemm_b_view()?;
            let out_all = subview_all(&out, hkv * group * s_len, d)?;
            for t in [&q_a, &k_b, &v_b, &out_all] {
                stream.retain_tensor(t);
            }
            if use_flash {
                let need = flash_attn_scratch_len(hq, s_len, t_len, d);
                if self.flash_scratch.as_ref().is_none_or(|t| t.len() < need) {
                    self.flash_scratch = Some(unsafe {
                        DeviceTensor::uninitialized_dt(
                            f32::datum_type(),
                            &[(need * 2).next_power_of_two()],
                        )?
                    });
                }
                return dispatch_gpt_oss_flash_attn_f16(
                    stream,
                    &q_a,
                    &k_b,
                    &v_b,
                    &mask_2d,
                    &sinks_flat,
                    &out_all,
                    self.flash_scratch.as_ref().unwrap(),
                    GptOssFlashAttnDims {
                        hq,
                        s_len,
                        t_len,
                        d,
                        group,
                        k_head_stride: self.k.cap * d,
                        v_head_stride: self.v.cap * d,
                        v_seq_stride: self.v.cap,
                    },
                    self.scale,
                );
            }
            let scores = Arc::new(get_context()?.uninitialized_device_tensor(
                &[hkv, group * s_len, t_len],
                f16::datum_type(),
            )?);
            let probs = Arc::new(get_context()?.uninitialized_device_tensor(
                &[hkv, group * s_len, t_len],
                f16::datum_type(),
            )?);
            let scores_all = subview_all(&scores, hkv * group * s_len, t_len)?;
            let probs_all = subview_all(&probs, hkv * group * s_len, t_len)?;
            stream.retain_tensor(&scores_all);
            stream.retain_tensor(&probs_all);
            // One batched gemm across all kv heads: A [Hkv, group*S, D] x
            // K^T [Hkv, T, D] -> scores [Hkv, group*S, T].
            gemm.dispatch_eval(
                stream,
                GemmDispatchParams {
                    dts: [f16dt, f16dt, f16dt],
                    a_batch: hkv,
                    b_batch: hkv,
                    m: group * s_len,
                    k: d,
                    n: t_len,
                    transpose_a: false,
                    a_offset: q_a_offset,
                    transpose_b: true,
                    b_offset: k_b.buffer_offset(),
                    q40_b: false,
                    c_offset: 0,
                    a_strides: tvec![(group * s_len * d) as isize, d as isize, 1],
                    b_strides: tvec![(self.k.cap * d) as isize, d as isize, 1],
                },
                get_metal_buffer(&q_a),
                get_metal_buffer(&k_b),
                get_metal_buffer(&scores_all),
            )?;
            // Fused scale+mask+sinks softmax over all rows at once.
            dispatch_gpt_oss_sinks_softmax_f16(
                stream,
                &scores_all,
                &mask_2d,
                &sinks_flat,
                &probs_all,
                s_len,
                self.scale,
            )?;
            // One batched gemm: probs [Hkv, group*S, T] x V^T [Hkv, D, T]^T
            // -> out [Hkv, group*S, D]. V is stored transposed so its k axis
            // (the sequence) is contiguous, as the GGML kernels require.
            gemm.dispatch_eval(
                stream,
                GemmDispatchParams {
                    dts: [f16dt, f16dt, f16dt],
                    a_batch: hkv,
                    b_batch: hkv,
                    m: group * s_len,
                    k: t_len,
                    n: d,
                    transpose_a: false,
                    a_offset: 0,
                    transpose_b: true,
                    b_offset: v_b.buffer_offset(),
                    q40_b: false,
                    c_offset: 0,
                    a_strides: tvec![(group * s_len * t_len) as isize, t_len as isize, 1],
                    b_strides: tvec![(self.v.d * self.v.cap) as isize, self.v.cap as isize, 1],
                },
                get_metal_buffer(&probs_all),
                get_metal_buffer(&v_b),
                get_metal_buffer(&out_all),
            )?;
            Ok(())
        })?;

        if std::env::var_os("TRACT_DEBUG_GPT_OSS_SELFCHECK").is_some() {
            crate::with_metal_stream(|stream| stream.wait_until_completed())?;
            // Rerun this step's attention on the CPU op from the SAME inputs
            // and compare: separates wrong-inputs from wrong-compute.
            use tract_transformers::ops::gpt_oss_sdpa::GptOssSdpa as CpuOp;
            let cpu_op = CpuOp { scale_bits: self.scale.to_bits() };
            let mut cpu_state = tract_core::ops::EvalOp::state(&cpu_op, _state, 0)?.unwrap();
            let host = |t: &DeviceTensor| -> TractResult<TValue> {
                Ok(t.to_host()?.into_tensor().into_tvalue())
            };
            let cpu_out = cpu_state.eval(
                _state,
                &cpu_op,
                tvec!(
                    host(q)?,
                    host(k_new)?,
                    host(v_new)?,
                    host(k_cache)?,
                    host(v_cache)?,
                    host(mask)?,
                    host(sinks)?,
                ),
            )?;
            let metal_out = DeviceTensor::ArenaView(DeviceArenaView::from_owned(
                out.clone(),
                f16::datum_type(),
                tvec![1, hq, s_len, d],
                natural_strides(&[1, hq, s_len, d]),
                0,
            )?)
            .to_host()?
            .into_tensor()
            .cast_to::<f32>()?
            .into_owned();
            let want = cpu_out[0].clone().into_tensor().cast_to::<f32>()?.into_owned();
            let mv = metal_out.try_as_plain()?.as_slice::<f32>()?;
            let wv = want.try_as_plain()?.as_slice::<f32>()?;
            let dot: f32 = mv.iter().zip(wv).map(|(a, b)| a * b).sum();
            let nm: f32 = mv.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nw: f32 = wv.iter().map(|a| a * a).sum::<f32>().sqrt();
            let max_abs = mv
                .iter()
                .zip(wv)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "gptoss-selfcheck: cosine {:.6} max_abs {:.4} (norms {:.2}/{:.2})",
                dot / (nm * nw).max(f32::MIN_POSITIVE),
                max_abs,
                nm,
                nw
            );
        }
        if std::env::var_os("TRACT_DEBUG_GPT_OSS_SDPA").is_some() {
            crate::with_metal_stream(|stream| stream.wait_until_completed())?;
            let stats = |t: &DeviceTensor, tag: &str| -> TractResult<()> {
                let host = t.to_host()?.into_tensor().cast_to::<f32>()?.into_owned();
                let v = host.try_as_plain()?.as_slice::<f32>()?;
                let nan = v.iter().filter(|x| x.is_nan()).count();
                let inf = v.iter().filter(|x| x.is_infinite()).count();
                let mx = v.iter().cloned().fold(f32::MIN, f32::max);
                let mn = v.iter().cloned().fold(f32::MAX, f32::min);
                let sum: f32 = v.iter().sum();
                eprintln!(
                    "gptoss-dbg {tag}: shape={:?} min={mn:.4} max={mx:.4} mean={:.5} nan={nan} inf={inf}",
                    t.shape(),
                    sum / v.len() as f32
                );
                Ok(())
            };
            let in_stats = |t: &DeviceTensor, tag: &str| -> TractResult<()> {
                let h = t.to_host()?.into_tensor().cast_to::<f32>()?.into_owned();
                let v = h.try_as_plain()?.as_slice::<f32>()?;
                let mx = v.iter().cloned().fold(f32::MIN, f32::max);
                let mn = v.iter().cloned().fold(f32::MAX, f32::min);
                let sum: f32 = v.iter().sum();
                eprintln!(
                    "gptoss-dbg {tag}: shape={:?} min={mn:.4} max={mx:.4} mean={:.6}",
                    t.shape(),
                    sum / v.len() as f32
                );
                Ok(())
            };
            in_stats(q, "q_in")?;
            in_stats(k_new, "k_new_in")?;
            stats(&self.k.valid_view()?, "k_cache")?;
            stats(&self.v.valid_view()?, "v_cache")?;
            stats(&subview_all(&out, hkv * group * s_len, d)?, "out")?;
            stats(&mask_2d, "mask")?;
            let sv = sinks_flat.to_host()?.into_tensor();
            let sv = sv.cast_to::<f32>()?.into_owned();
            eprintln!(
                "gptoss-dbg sinks[0..6]: {:?}",
                &sv.try_as_plain()?.as_slice::<f32>()?[..6.min(hq)]
            );
        }
        let out_t = DeviceTensor::ArenaView(DeviceArenaView::from_owned(
            out,
            f16::datum_type(),
            tvec![1, hq, s_len, d],
            natural_strides(&[1, hq, s_len, d]),
            0,
        )?);
        Ok(tvec!(
            out_t.into_tensor().into_tvalue(),
            self.k.valid_view()?.into_tensor().into_tvalue(),
            self.v.valid_view()?.into_tensor().into_tvalue(),
        ))
    }
}

fn subview_all(
    arc: &Arc<Box<dyn OwnedDeviceTensor>>,
    rows: usize,
    cols: usize,
) -> TractResult<DeviceTensor> {
    Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
        arc.clone(),
        f16::datum_type(),
        tvec![rows, cols],
        natural_strides(&[rows, cols]),
        0,
    )?))
}

/// [1, rows, cols] view of head `h` in a dense [Hkv, rows, cols] tensor.
fn subview_head(
    arc: &Arc<Box<dyn OwnedDeviceTensor>>,
    rows: usize,
    cols: usize,
    h: usize,
) -> TractResult<DeviceTensor> {
    Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
        arc.clone(),
        f16::datum_type(),
        tvec![1, rows, cols],
        natural_strides(&[1, rows, cols]),
        h * rows * cols * f16::datum_type().size_of(),
    )?))
}

use tract_core::ops::{FrozenOpState, OpStateFreeze};

#[derive(Clone, Debug)]
pub struct FrozenMetalGptOssSdpaState(MetalGptOssSdpaState);

impl OpStateFreeze for MetalGptOssSdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenMetalGptOssSdpaState(self.clone()))
    }
}

impl FrozenOpState for FrozenMetalGptOssSdpaState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(self.0.clone())
    }
}

crate::register_metal_op!(GptOssSdpa, |_source, _node, op| {
    if std::env::var_os("TRACT_METAL_DISABLE_GPT_OSS_SDPA").is_some() {
        return Ok(None);
    }
    Ok(Some(Box::new(MetalGptOssSdpa { scale_bits: op.scale_bits })))
});

#[cfg(test)]
mod tests {
    use super::*;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::gpt_oss_sdpa::GptOssSdpa as CpuOp;

    fn rng_tensor(shape: &[usize], seed: &mut u64) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        Tensor::from_shape(shape, &data).unwrap()
    }

    fn causal_mask(s_len: usize, kv_len: usize) -> Tensor {
        window_mask(s_len, kv_len, usize::MAX)
    }

    /// Causal mask with a sliding window: key j visible to query (past+i)
    /// iff j <= past+i and past+i - j < window.
    fn window_mask(s_len: usize, kv_len: usize, window: usize) -> Tensor {
        let past = kv_len - s_len;
        let mut m = Tensor::zero::<f32>(&[1, 1, s_len, kv_len]).unwrap();
        {
            let mut v = m.to_plain_array_view_mut::<f32>().unwrap();
            for i in 0..s_len {
                for j in 0..kv_len {
                    let q = past + i;
                    if j > q || (window != usize::MAX && q - j >= window) {
                        v[[0, 0, i, j]] = -9.2e18;
                    }
                }
            }
        }
        m
    }

    #[test]
    fn metal_matches_cpu_real_geometry_prefill() -> TractResult<()> {
        run_metal_vs_cpu(64, 8, 64, &[256, 1, 1], usize::MAX)
    }

    /// 12 of 24 GPT-OSS layers run a 128-key sliding-window mask, which only
    /// differs from causal past ~128 tokens of context: exactly where the
    /// in-graph Metal corruption starts.
    #[test]
    fn metal_matches_cpu_sliding_window() -> TractResult<()> {
        run_metal_vs_cpu(64, 8, 64, &[256, 1, 1, 40], 128)
    }

    #[test]
    fn metal_matches_cpu_over_prefill_and_decode() -> TractResult<()> {
        run_metal_vs_cpu(4, 2, 64, &[5, 1, 1, 1500, 1], usize::MAX)
    }

    /// Transposed capacity buffer: append twice, read back through the
    /// logical view, expect the concatenation.
    #[test]
    fn transposed_kv_buffer_roundtrip() -> TractResult<()> {
        crate::context::metal_context();
        let (hkv, d) = (2usize, 4usize);
        let mut buf = DeviceKvBuffer::transposed();
        let mut seed = 7u64;
        let c1 = rng_tensor(&[1, hkv, 3, d], &mut seed).cast_to::<f16>()?.into_owned();
        let c2 = rng_tensor(&[1, hkv, 2, d], &mut seed).cast_to::<f16>()?.into_owned();
        crate::with_metal_stream(|stream| {
            buf.append(&c1.clone().into_device()?)?;
            buf.append(&c2.clone().into_device()?)?;
            stream.wait_until_completed()
        })?;
        let got = buf.valid_view()?.to_host()?.into_tensor();
        let want = Tensor::stack_tensors(2, &[&c1, &c2])?;
        got.close_enough(&want, Approximation::Exact)?;
        Ok(())
    }

    /// Prefix-cache truncation: feed back the op's own device cache views
    /// sliced (metadata-only) to a shorter length, as causal_llm's truncate
    /// does on a session restore. The op must rebuild into a fresh buffer,
    /// keep matching the CPU reference, and leave the longer snapshot's
    /// bytes untouched (other sessions may still hold views of it).
    #[test]
    fn metal_truncation_matches_cpu() -> TractResult<()> {
        use tract_gpu::tensor::DeviceTensor;
        force_flash_for_tests();
        crate::context::metal_context();
        let (hq, hkv, d) = (4usize, 2usize, 64usize);
        let scale = (d as f32).sqrt().recip();
        let op = MetalGptOssSdpa { scale_bits: scale.to_bits() };
        let cpu_op = CpuOp { scale_bits: scale.to_bits() };
        let mut session = TurnState::default();
        let mut metal_state = op.state(&session, 0)?.unwrap();
        let mut cpu_state = EvalOp::state(&cpu_op, &session, 0)?.unwrap();

        let mut seed = 43u64;
        let sinks = rng_tensor(&[hq], &mut seed);
        let mut k_all = Tensor::zero::<f16>(&[1, hkv, 0, d])?;
        let mut v_all = Tensor::zero::<f16>(&[1, hkv, 0, d])?;
        let mut metal_k = Tensor::zero::<f16>(&[1, hkv, 0, d])?.into_device()?.into_tensor().into_tvalue();
        let mut metal_v = Tensor::zero::<f16>(&[1, hkv, 0, d])?.into_device()?.into_tensor().into_tvalue();

        // (step_len, truncate-to-before-step)
        let plan: &[(usize, Option<usize>)] =
            &[(8, None), (1, None), (1, None), (1, Some(6)), (1, None)];
        let mut snapshot: Option<(TValue, Tensor)> = None;
        for &(step_len, trunc) in plan {
            if let Some(cols) = trunc {
                // Keep the long snapshot to check it survives the overwrite.
                snapshot = Some((metal_k.clone(), k_all.clone()));
                let slice_dev = |t: &TValue| -> TractResult<TValue> {
                    let DeviceTensor::ArenaView(view) = t.to_device_tensor()? else {
                        bail!("expected an arena view cache output")
                    };
                    Ok(DeviceTensor::ArenaView(view.sliced(2, 0, cols)?)
                        .into_tensor()
                        .into_tvalue())
                };
                metal_k = slice_dev(&metal_k)?;
                metal_v = slice_dev(&metal_v)?;
                k_all = k_all.slice(2, 0, cols)?;
                v_all = v_all.slice(2, 0, cols)?;
            }
            let past = k_all.shape()[2];
            let q = rng_tensor(&[1, hq, step_len, d], &mut seed).cast_to::<f16>()?.into_owned();
            let k_new =
                rng_tensor(&[1, hkv, step_len, d], &mut seed).cast_to::<f16>()?.into_owned();
            let v_new =
                rng_tensor(&[1, hkv, step_len, d], &mut seed).cast_to::<f16>()?.into_owned();
            let mask = window_mask(step_len, past + step_len, usize::MAX);

            let cpu_out = cpu_state.eval(
                &mut session,
                &cpu_op,
                tvec!(
                    q.clone().into_tvalue(),
                    k_new.clone().into_tvalue(),
                    v_new.clone().into_tvalue(),
                    k_all.clone().into_tvalue(),
                    v_all.clone().into_tvalue(),
                    mask.clone().into_tvalue(),
                    sinks.clone().into_tvalue(),
                ),
            )?;
            let metal_out = metal_state.eval(
                &mut session,
                &op,
                tvec!(
                    q.clone().into_device()?.into_tensor().into_tvalue(),
                    k_new.clone().into_device()?.into_tensor().into_tvalue(),
                    v_new.clone().into_device()?.into_tensor().into_tvalue(),
                    metal_k.clone(),
                    metal_v.clone(),
                    mask.clone().into_device()?.into_tensor().into_tvalue(),
                    sinks.clone().into_device()?.into_tensor().into_tvalue(),
                ),
            )?;
            crate::with_metal_stream(|stream| stream.wait_until_completed())?;

            let got = metal_out[0].to_device_tensor()?.to_host()?.into_tensor();
            let want = cpu_out[0].clone().into_tensor();
            got.cast_to::<f32>()?
                .into_owned()
                .close_enough(&want.cast_to::<f32>()?.into_owned(), Approximation::SuperApproximate)
                .with_context(|| format!("truncation test output at past={past}"))?;
            for i in [1usize, 2] {
                let got = metal_out[i].to_device_tensor()?.to_host()?.into_tensor();
                let want = cpu_out[i].clone().into_tensor();
                got.close_enough(&want, Approximation::Exact)
                    .with_context(|| format!("truncation test cache {i} at past={past}"))?;
            }
            k_all = cpu_out[1].clone().into_tensor();
            v_all = cpu_out[2].clone().into_tensor();
            metal_k = metal_out[1].clone();
            metal_v = metal_out[2].clone();
        }
        // The pre-truncation snapshot view must still read its original bytes.
        let (snap_dev, snap_host) = snapshot.unwrap();
        let snap_read = snap_dev.to_device_tensor()?.to_host()?.into_tensor();
        snap_read
            .close_enough(&snap_host, Approximation::Exact)
            .context("pre-truncation cache snapshot was corrupted by later appends")?;
        Ok(())
    }

    /// Force decode steps onto the flash path regardless of context length,
    /// so the CPU-comparison tests exercise it at test-sized T. All tests
    /// setting this use the same value, so parallel execution is safe; the
    /// gemm decode path is covered by the TRACT_METAL_DISABLE_GPT_OSS_FLASH
    /// suite run.
    fn force_flash_for_tests() {
        unsafe { std::env::set_var("TRACT_METAL_GPT_OSS_FLASH_MIN_T", "0") };
    }

    fn run_metal_vs_cpu(
        hq: usize,
        hkv: usize,
        d: usize,
        steps: &[usize],
        window: usize,
    ) -> TractResult<()> {
        force_flash_for_tests();
        crate::context::metal_context(); // initialize the GPU context
        let scale = (d as f32).sqrt().recip();
        let op = MetalGptOssSdpa { scale_bits: scale.to_bits() };
        let cpu_op = CpuOp { scale_bits: scale.to_bits() };
        let mut session = TurnState::default();
        let mut metal_state = op.state(&session, 0)?.unwrap();
        let mut cpu_state = EvalOp::state(&cpu_op, &session, 0)?.unwrap();

        let mut seed = 42u64;
        let sinks = rng_tensor(&[hq], &mut seed);
        let mut k_all = Tensor::zero::<f16>(&[1, hkv, 0, d])?;
        let mut v_all = Tensor::zero::<f16>(&[1, hkv, 0, d])?;

        for &step_len in steps {
            let past = k_all.shape()[2];
            let q = rng_tensor(&[1, hq, step_len, d], &mut seed).cast_to::<f16>()?.into_owned();
            let k_new =
                rng_tensor(&[1, hkv, step_len, d], &mut seed).cast_to::<f16>()?.into_owned();
            let v_new =
                rng_tensor(&[1, hkv, step_len, d], &mut seed).cast_to::<f16>()?.into_owned();
            let mask = window_mask(step_len, past + step_len, window);

            let cpu_out = cpu_state.eval(
                &mut session,
                &cpu_op,
                tvec!(
                    q.clone().into_tvalue(),
                    k_new.clone().into_tvalue(),
                    v_new.clone().into_tvalue(),
                    k_all.clone().into_tvalue(),
                    v_all.clone().into_tvalue(),
                    mask.clone().into_tvalue(),
                    sinks.clone().into_tvalue(),
                ),
            )?;

            let metal_out = metal_state.eval(
                &mut session,
                &op,
                tvec!(
                    q.clone().into_device()?.into_tensor().into_tvalue(),
                    k_new.clone().into_device()?.into_tensor().into_tvalue(),
                    v_new.clone().into_device()?.into_tensor().into_tvalue(),
                    k_all.clone().into_device()?.into_tensor().into_tvalue(),
                    v_all.clone().into_device()?.into_tensor().into_tvalue(),
                    mask.clone().into_device()?.into_tensor().into_tvalue(),
                    sinks.clone().into_device()?.into_tensor().into_tvalue(),
                ),
            )?;
            crate::with_metal_stream(|stream| stream.wait_until_completed())?;

            for (i, tol) in [(0usize, Approximation::SuperApproximate)].into_iter() {
                let got = metal_out[i].to_device_tensor()?.to_host()?.into_tensor();
                let want = cpu_out[i].clone().into_tensor();
                got.cast_to::<f32>()?
                    .into_owned()
                    .close_enough(&want.cast_to::<f32>()?.into_owned(), tol)
                    .with_context(|| format!("step past={past} output {i}"))?;
            }
            // caches: exact f16 equality vs cpu emission
            for i in [1usize, 2] {
                let got = metal_out[i].to_device_tensor()?.to_host()?.into_tensor();
                let want = cpu_out[i].clone().into_tensor();
                got.close_enough(&want, Approximation::Exact)
                    .with_context(|| format!("step past={past} cache {i}"))?;
            }

            k_all = cpu_out[1].clone().into_tensor();
            v_all = cpu_out[2].clone().into_tensor();
        }
        Ok(())
    }
}
