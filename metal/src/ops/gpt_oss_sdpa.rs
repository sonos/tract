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

use crate::kernels::matmul::{BasicMatMul, GemmImpl, GgmlGemm};
use crate::kernels::moe::dispatch_gpt_oss_sinks_softmax_f16;

const SEQ_AXIS: usize = 2;

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
            v: DeviceKvBuffer::default(),
        })))
    }
}

/// One device-side capacity buffer, [1, Hkv, cap, D] f16.
#[derive(Clone, Default)]
struct DeviceKvBuffer {
    buf: Option<Arc<Box<dyn OwnedDeviceTensor>>>,
    hkv: usize,
    d: usize,
    cap: usize,
    len: usize,
}

impl std::fmt::Debug for DeviceKvBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DeviceKvBuffer(len={}, cap={})", self.len, self.cap)
    }
}

impl DeviceKvBuffer {
    fn full_view(&self) -> TractResult<DeviceTensor> {
        let buf = self.buf.as_ref().context("empty kv buffer")?;
        Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
            buf.clone(),
            f16::datum_type(),
            tvec![1, self.hkv, self.cap, self.d],
            natural_strides(&[1, self.hkv, self.cap, self.d]),
            0,
        )?))
    }

    /// Valid region as a strided view [1, Hkv, len, D] (per-head stride cap*D).
    fn valid_view(&self) -> TractResult<DeviceTensor> {
        let buf = self.buf.as_ref().context("empty kv buffer")?;
        Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
            buf.clone(),
            f16::datum_type(),
            tvec![1, self.hkv, self.len, self.d],
            natural_strides(&[1, self.hkv, self.cap, self.d]),
            0,
        )?))
    }

    /// Contiguous [T, D] view of one head's valid region.
    fn head_view(&self, h: usize) -> TractResult<DeviceTensor> {
        let buf = self.buf.as_ref().context("empty kv buffer")?;
        Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
            buf.clone(),
            f16::datum_type(),
            tvec![1, self.len, self.d],
            natural_strides(&[1, self.len, self.d]),
            h * self.cap * self.d * f16::datum_type().size_of(),
        )?))
    }

    fn alloc(hkv: usize, d: usize, cap: usize) -> TractResult<Arc<Box<dyn OwnedDeviceTensor>>> {
        Ok(Arc::new(
            get_context()?.uninitialized_device_tensor(&[1, hkv, cap, d], f16::datum_type())?,
        ))
    }

    /// Append `chunk` ([1, Hkv, new, D] f16 device tensor) past `len`.
    fn append(&mut self, chunk: &DeviceTensor) -> TractResult<()> {
        let shape = chunk.shape();
        ensure!(shape.len() == 4 && shape[0] == 1);
        let (hkv, new, d) = (shape[1], shape[2], shape[3]);
        let ctx = get_context()?;
        if self.buf.is_none() {
            let cap = (new * 2).max(256);
            self.buf = Some(Self::alloc(hkv, d, cap)?);
            self.hkv = hkv;
            self.d = d;
            self.cap = cap;
            self.len = 0;
        }
        ensure!(hkv == self.hkv && d == self.d, "kv geometry changed");
        if self.len + new > self.cap {
            let new_cap = (self.cap * 2).max(self.len + new);
            let grown = Self::alloc(hkv, d, new_cap)?;
            let old_valid = self.valid_view()?;
            let grown_view = DeviceTensor::ArenaView(DeviceArenaView::from_owned(
                grown.clone(),
                f16::datum_type(),
                tvec![1, hkv, self.len, d],
                natural_strides(&[1, hkv, new_cap, d]),
                0,
            )?);
            ctx.copy_nd(
                &old_valid,
                0,
                old_valid.strides(),
                &grown_view,
                0,
                &tvec![1, hkv, self.len, d],
                grown_view.strides(),
            )?;
            self.buf = Some(grown);
            self.cap = new_cap;
        }
        let dst = self.full_view()?;
        let dst_offset =
            self.len * self.d * f16::datum_type().size_of();
        ctx.copy_nd(
            chunk,
            0,
            chunk.strides(),
            &dst,
            dst_offset,
            &tvec![1, hkv, new, d],
            dst.strides(),
        )?;
        self.len += new;
        Ok(())
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Debug)]
pub struct MetalGptOssSdpaState {
    scale: f32,
    k: DeviceKvBuffer,
    v: DeviceKvBuffer,
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

        // Scores/probs/output scratch.
        let scores = Arc::new(get_context()?.uninitialized_device_tensor(
            &[hkv, group * s_len, t_len],
            f16::datum_type(),
        )?);
        let probs = Arc::new(get_context()?.uninitialized_device_tensor(
            &[hkv, group * s_len, t_len],
            f16::datum_type(),
        )?);
        let out = Arc::new(get_context()?.uninitialized_device_tensor(
            &[hkv, group * s_len, d],
            f16::datum_type(),
        )?);

        let subview = |arc: &Arc<Box<dyn OwnedDeviceTensor>>,
                       rows: usize,
                       cols: usize,
                       head: usize|
         -> TractResult<DeviceTensor> {
            Ok(DeviceTensor::ArenaView(DeviceArenaView::from_owned(
                arc.clone(),
                f16::datum_type(),
                tvec![1, rows, cols],
                natural_strides(&[1, rows, cols]),
                head * rows * cols * f16::datum_type().size_of(),
            )?))
        };

        // q is copied once into op-owned scratch so per-head offset views can
        // Arc-share it (a few KB per decode step).
        let q_scratch = Arc::new(get_context()?.uninitialized_device_tensor(
            &[hkv, group * s_len, d],
            f16::datum_type(),
        )?);
        {
            let ctx = get_context()?;
            let dst = subview_all(&q_scratch, hkv * group * s_len, d)?;
            let q_flat = q.reshaped(tvec![hkv * group * s_len, d])?;
            ctx.copy_nd(
                &q_flat,
                0,
                q_flat.strides(),
                &dst,
                0,
                &tvec![hkv * group * s_len, d],
                dst.strides(),
            )?;
        }
        let qk = GemmImpl { transpose_a: false, transpose_b: true, matmul: GgmlGemm };
        // ggml kernels require B as [n,k] (transpose_b); AV multiplies by V in
        // [k,n] layout, so it runs on the general BasicMatMul kernel instead.
        let av = GemmImpl { transpose_a: false, transpose_b: false, matmul: BasicMatMul };

        crate::with_metal_stream(|stream| {
            // Command-buffer boundary between the cache appends (copy_nd
            // above) and the QK gemms reading the same buffer: the Metal
            // runtime's intra-buffer write->read visibility defect corrupts
            // this exact pattern past ~1024 tokens (same medicine as the MoE
            // routed-matmul splits).
            stream.commit_current()?;
            for h in 0..hkv {
                let q_h = subview(&q_scratch, group * s_len, d, h)?;
                let k_h = self.k.head_view(h)?;
                let s_h = subview(&scores, group * s_len, t_len, h)?;
                qk.dispatch_eval(stream, &q_h, &k_h, &s_h)?;
            }
            // Fused scale+mask+sinks softmax over all rows at once.
            let scores_all = subview_all(&scores, hkv * group * s_len, t_len)?;
            let probs_all = subview_all(&probs, hkv * group * s_len, t_len)?;
            dispatch_gpt_oss_sinks_softmax_f16(
                stream,
                &scores_all,
                &mask_2d,
                &sinks_flat,
                &probs_all,
                s_len,
                self.scale,
            )?;
            for h in 0..hkv {
                let p_h = subview(&probs, group * s_len, t_len, h)?;
                let v_h = self.v.head_view(h)?;
                let o_h = subview(&out, group * s_len, d, h)?;
                av.dispatch_eval(stream, &p_h, &v_h, &o_h)?;
            }
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
            stats(&subview_all(&q_scratch, hkv * group * s_len, d)?, "q_scratch")?;
            stats(&self.k.valid_view()?, "k_cache")?;
            stats(&subview_all(&scores, hkv * group * s_len, t_len)?, "scores")?;
            stats(&subview_all(&probs, hkv * group * s_len, t_len)?, "probs")?;
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

    fn run_metal_vs_cpu(
        hq: usize,
        hkv: usize,
        d: usize,
        steps: &[usize],
        window: usize,
    ) -> TractResult<()> {
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
