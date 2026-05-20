//! Fused GRU op (A2).
//!
//! Replaces the Scan-with-decomposed-body lowering produced by
//! `onnx/src/ops/rec/gru.rs::wire_body` when attribute combinations are
//! supported. Avoids the per-timestep `SimplePlan` re-entry overhead
//! (~9 µs/iter on Apple Silicon native) measured in
//! `ort-web-vs-tract-investigation.md` §3 RC1.
//!
//! The recurrence runs in a single Rust loop inside `eval`: one matmul per
//! gate per timestep, fused gate elementwise, no per-node `TVec` allocs, no
//! plan re-entry.
//!
//! Phase 1 of A2: correctness-first using `ndarray::dot` for matmuls + scalar
//! sigmoid/tanh. Phase 2 will swap in `tract_linalg::ops().mmm()` directly to
//! match OptMatMul kernel quality, plus the linalg vectorized sigmoid/tanh
//! installed by PR #2195.

use crate::internal::*;
use ndarray::*;
use std::sync::Arc;
use tract_linalg::mmm::{AsInputValue, EagerPackedInput, FusedSpec, MMMInputValue};

/// A constant GRU weight (`W` or `R`) packed once for the linalg matmul kernel,
/// cached on the op state across inferences.
type PackedWeight = Option<Box<dyn MMMInputValue>>;

/// Fused GRU op covering one direction.
///
/// Inputs (5):
///   0: X       [seq_len, batch, input_size]    sequence
///   1: W       [3*hidden, input_size]          input weights (Wz, Wr, Wh concat)
///   2: R       [3*hidden, hidden]              recurrent weights (Rz, Rr, Rh concat)
///   3: B       [6*hidden]                      biases (Wbz, Wbr, Wbh, Rbz, Rbr, Rbh)
///   4: h0      [batch, hidden]                 initial hidden state
///
/// Outputs (2) — emitted in **Scan-body-accumulator layout** so the same
/// post-Scan wire ops (Move + Add for !batch_first) can be applied. This
/// matches what `tract_core::ops::scan::Scan` produces, letting the optimizer
/// see identical downstream-op structure for both lowering paths.
///   0: Y       [batch, seq_len, hidden]        full sequence of hidden states
///   1: Y_h     [batch, 1, hidden]              final hidden state (with chunk axis)
///
/// Bidirectional GRUs use two `OptGru` ops + a `Concat`. The caller (ONNX
/// import in `onnx/src/ops/rec/common.rs`) is responsible for direction
/// slicing, batch_first transposes, and materializing zero tensors for the
/// optional bias / initial-h ONNX inputs when they are absent.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OptGru {
    pub hidden_size: usize,
    pub input_size: usize,
    /// ONNX `linear_before_reset` attribute. `true` is the PyTorch default
    /// and the configuration used by DFN3.
    pub linear_before_reset: bool,
}

impl Op for OptGru {
    fn name(&self) -> StaticName {
        "OptGru".to_string().into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "hidden={}, input={}, linear_before_reset={}",
            self.hidden_size, self.input_size, self.linear_before_reset,
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for OptGru {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // Direct (stateless) call path — unit tests / const-eval. No persistent
        // weight cache: the W/R weights are packed fresh each call.
        let (mut wc, mut rc): (PackedWeight, PackedWeight) = (None, None);
        self.run_eval(inputs, &mut wc, &mut rc)
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        // Stateful so the packed recurrent-weight matrix R is computed ONCE and
        // reused across inferences. Packing R[3*hidden, hidden] (~786 KB) costs
        // ~1.7 ms on wasm; re-doing it per call dominated the op (the
        // recurrence loop itself is only ~1.4 ms). R is a constant GRU weight.
        Ok(Some(Box::new(OptGruState::default())))
    }
}

impl OptGru {
    fn run_eval(
        &self,
        inputs: TVec<TValue>,
        packed_w_cache: &mut PackedWeight,
        packed_r_cache: &mut PackedWeight,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 5, "OptGru expects 5 inputs (X, W, R, B, h0)");

        let dt = inputs[0].datum_type();
        let x = inputs[0].cast_to::<f32>()?.into_owned();
        let w = inputs[1].cast_to::<f32>()?.into_owned();
        let r = inputs[2].cast_to::<f32>()?.into_owned();
        let b = inputs[3].cast_to::<f32>()?.into_owned();
        let h0 = inputs[4].cast_to::<f32>()?.into_owned();

        let x_view = x.to_plain_array_view::<f32>()?.into_dimensionality::<Ix3>()?;
        let w_view = w.to_plain_array_view::<f32>()?.into_dimensionality::<Ix2>()?;
        let r_view = r.to_plain_array_view::<f32>()?.into_dimensionality::<Ix2>()?;
        let b_view = b.to_plain_array_view::<f32>()?.into_dimensionality::<Ix1>()?;
        let h0_view = h0.to_plain_array_view::<f32>()?.into_dimensionality::<Ix2>()?;

        let seq_len = x_view.shape()[0];
        let batch = x_view.shape()[1];
        let hidden = self.hidden_size;
        let input_size = self.input_size;

        ensure!(x_view.shape()[2] == input_size, "X input_size mismatch");
        ensure!(w_view.shape() == [3 * hidden, input_size], "W shape mismatch");
        ensure!(r_view.shape() == [3 * hidden, hidden], "R shape mismatch");
        ensure!(b_view.shape() == [6 * hidden], "B shape mismatch");
        ensure!(h0_view.shape() == [batch, hidden], "h0 shape mismatch");

        // Slice biases and recurrent weights once outside the loop.
        let wbz = b_view.slice(s![0..hidden]);
        let wbr = b_view.slice(s![hidden..2 * hidden]);
        let wbh = b_view.slice(s![2 * hidden..3 * hidden]);
        let rbz = b_view.slice(s![3 * hidden..4 * hidden]);
        let rbr = b_view.slice(s![4 * hidden..5 * hidden]);
        let rbh = b_view.slice(s![5 * hidden..6 * hidden]);

        // Output Y in **Scan body accumulator** layout: [batch, seq, hidden].
        // Wire (in `onnx/src/ops/rec/gru.rs`) will then apply the same
        // Move(1,0) + Add(1) sequence Scan uses, letting the optimizer fuse
        // them into a single IntoShape — identical downstream-op chain for
        // both lowering paths.
        let mut y = Array3::<f32>::zeros((batch, seq_len, hidden));
        // Final hidden state, captured per-branch as [batch, hidden].
        let h_last: Array2<f32>;

        if self.linear_before_reset && batch == 1 {
            // ── FAST PATH (the DFN3 configuration: batch=1, lbr=1).
            // Everything runs on tract-linalg kernels (same as the Scan body),
            // so the only structural difference from Scan is the absence of
            // per-timestep plan re-entry.

            // (1) Hoisted input projection X·Wᵀ → [seq*batch, 3*hidden] via
            // tract-linalg `mmm`. ndarray's GEMM is markedly slower on wasm,
            // where this projection — NOT the recurrence — was OptGru's largest
            // per-call cost. W is a constant weight → packed ONCE and cached on
            // the op state. (Scoped in a block so the cache borrow is released
            // before the recurrent matmul borrows the R cache.)
            let mn = seq_len * batch;
            let xwt: Tensor = {
                let mmm_in = tract_linalg::ops()
                    .mmm(f32::datum_type(), Some(mn), Some(input_size), Some(3 * hidden))
                    .context("OptGru: no f32 mmm kernel for input projection")?;
                let (pack_a_in, pack_b_in) = &mmm_in.packings()[0];
                // W: [3*hidden, input] = [N, K] → pack as B (k_axis=1, mn_axis=0).
                if packed_w_cache.is_none() {
                    *packed_w_cache = Some(pack_b_in.prepare_one(&w, 1, 0)?);
                }
                let packed_w = packed_w_cache.as_ref().unwrap();
                // X: [seq*batch, input] = [M, K] → pack as A (k_axis=1, mn_axis=0).
                let x_2d: Tensor =
                    x_view.into_shape_with_order((mn, input_size))?.to_owned().into();
                let packed_x = pack_a_in.prepare_one(&x_2d, 1, 0)?;
                let mut xwt =
                    unsafe { Tensor::uninitialized_dt(f32::datum_type(), &[mn, 3 * hidden])? };
                unsafe {
                    let mut sc = mmm_in.allocate_scratch_space();
                    let c = mmm_in.c_view(Some(0), Some(1)).wrap(&xwt.view_mut());
                    mmm_in.run_with_scratch_space(
                        mn,
                        3 * hidden,
                        sc.as_mut(),
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*packed_x),
                                b: AsInputValue::Borrowed(&**packed_w),
                                packing: 0,
                            },
                            FusedSpec::Store(c),
                        ],
                    )?;
                }
                xwt
            };

            // (2) Per-timestep recurrent projection as the MMV `R · h_prev`: R
            // is stored `[3*hidden, hidden]` = `[M, K]`, h_prev is the `[K]`
            // vector, output `[M=3*hidden]` (N=1) → tract's `mmv_f32` kernel.
            // (Measured: this MMV orientation is ~4× faster than the transposed
            // M=1/N=3*hidden `mmm` form, whose M=1 wastes the kernel's m-tiling.)
            // R is constant → packed ONCE as the A operand and cached; only the
            // small h_prev vector is packed per timestep.
            let mmm = tract_linalg::ops()
                .mmm(f32::datum_type(), Some(3 * hidden), Some(hidden), Some(1))
                .context("OptGru fast path: no f32 mmm/mmv kernel available")?;
            let (pack_a, pack_b) = &mmm.packings()[0];
            // R: [3*hidden, hidden] = [M, K] → k_axis=1, mn_axis=0. Packed ONCE
            // and cached on the op state (R is a constant GRU weight): the pack
            // is ~786 KB and was the single most expensive step per call on wasm.
            if packed_r_cache.is_none() {
                *packed_r_cache = Some(pack_a.prepare_one(&r, 1, 0)?);
            }
            let packed_r = packed_r_cache.as_ref().unwrap();
            let mut scratch = unsafe { mmm.allocate_scratch_space() };
            // Reused output buffer for R · h_prev → [3*hidden, 1].
            let mut h_r = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &[3 * hidden, 1])? };

            // Vectorized SIMD sigmoid/tanh — the SAME kernels the Scan body
            // uses (WasmSigmoid4Relaxed/WasmTanh4Relaxed on wasm; NEON/AVX
            // equivalents natively). Scalar `(-x).exp()` / `f32::tanh()` are
            // software libm on wasm and dominate the per-timestep cost there;
            // the kernels are polynomial+SIMD. Built once, reused per timestep.
            let sigmoid = (tract_linalg::ops().sigmoid_f32)();
            let tanh = (tract_linalg::ops().tanh_f32)();

            // Bias slices as flat &[f32] (contiguous views into B).
            let wbz_s = wbz.as_slice().context("wbz not contiguous")?;
            let wbr_s = wbr.as_slice().context("wbr not contiguous")?;
            let wbh_s = wbh.as_slice().context("wbh not contiguous")?;
            let rbz_s = rbz.as_slice().context("rbz not contiguous")?;
            let rbr_s = rbr.as_slice().context("rbr not contiguous")?;
            let rbh_s = rbh.as_slice().context("rbh not contiguous")?;
            // Input projection as a flat slice: row t is
            // xwt_flat[t*3h .. (t+1)*3h] = [xt_z | xt_r | xt_h].
            let xwt_av = xwt.to_plain_array_view::<f32>()?;
            let xwt_flat = xwt_av.as_slice().context("xwt not contiguous")?;

            // Preallocated per-timestep scratch — the loop performs NO heap
            // allocation (critical on wasm, where dlmalloc is slow; Scan reuses
            // its plan buffers similarly). Buffers are reused every timestep.
            let mut h_prev: Vec<f32> = h0_view.row(0).to_vec();
            let mut new_h: Vec<f32> = vec![0.0; hidden];
            let mut zt: Vec<f32> = vec![0.0; hidden];
            let mut rt: Vec<f32> = vec![0.0; hidden];
            let mut ht: Vec<f32> = vec![0.0; hidden];
            let y_flat = y.as_slice_mut().context("y not contiguous")?;

            // Pack the B (activation) operand ONCE, then refresh its packed
            // buffer in place each timestep. The ONNX GRU recurrence is
            // sequential, so h_prev must be packed every step — but
            // `prepare_one` allocates a Blob + Arc + Box per call, and on wasm
            // (slow dlmalloc) that allocation cost more than the plan re-entry
            // OptGru removes, leaving OptGru at mere parity with Scan. For the
            // mmv kernel the B packing format has r()==1, so packing the [K,1]
            // vector is a *contiguous memcpy* of K f32 (see
            // `PackedFormat::pack_t`). We therefore pack once and overwrite
            // those K floats per timestep with zero further allocation.
            let mut h_prev_t = Tensor::zero::<f32>(&[1, hidden])?;
            unsafe { h_prev_t.as_slice_mut_unchecked::<f32>() }.copy_from_slice(&h_prev);
            ensure!(
                pack_b.r() == 1,
                "OptGru fast path requires an mmv (nr==1) B packing; got r()={}",
                pack_b.r()
            );
            let mut packed_h: Box<EagerPackedInput> = pack_b
                .prepare_one(&h_prev_t, 1, 0)?
                .downcast::<EagerPackedInput>()
                .ok()
                .context("OptGru: B packing is not an EagerPackedInput")?;
            // Zero once so any k-alignment padding stays 0 across timesteps
            // (the per-step memcpy only refreshes the first `hidden` floats).
            if let Some(blob) = Arc::get_mut(&mut packed_h.packed) {
                blob.as_bytes_mut().fill(0u8);
            }

            for t in 0..seq_len {
                // Refresh the packed B buffer with the current h_prev — a plain
                // memcpy of K=hidden f32 into the uniquely-owned packed Blob, no
                // allocation. SAFETY: the Arc is never cloned (FusedSpec only
                // borrows &packed_h) so refcount stays 1 and `get_mut` succeeds;
                // the blob holds >= hidden contiguous f32 (mmv r()==1 layout).
                {
                    let blob = Arc::get_mut(&mut packed_h.packed)
                        .context("OptGru: packed B buffer unexpectedly shared")?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            h_prev.as_ptr(),
                            blob.as_mut_ptr() as *mut f32,
                            hidden,
                        );
                    }
                }
                unsafe {
                    let c = mmm.c_view(Some(0), Some(1)).wrap(&h_r.view_mut());
                    mmm.run_with_scratch_space(
                        3 * hidden,
                        1,
                        scratch.as_mut(),
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&**packed_r),
                                b: AsInputValue::Borrowed(&*packed_h),
                                packing: 0,
                            },
                            FusedSpec::Store(c),
                        ],
                    )?;
                }
                // h_r = [h_rz | h_rr | h_rh]; x row = [xt_z | xt_r | xt_h].
                // Slice both into exact-length thirds so the per-element loops
                // are bounds-check-free and vectorize (NEON / simd128 fmla).
                let h_r_av = h_r.to_plain_array_view::<f32>()?;
                let h_r_s = h_r_av.as_slice().context("h_r not contiguous")?;
                let (hrz, hrr, hrh) =
                    (&h_r_s[0..hidden], &h_r_s[hidden..2 * hidden], &h_r_s[2 * hidden..]);
                let row = &xwt_flat[t * 3 * hidden..(t + 1) * 3 * hidden];
                let (xz, xr, xh) = (&row[0..hidden], &row[hidden..2 * hidden], &row[2 * hidden..]);

                // z, r gates (sigmoid), candidate (tanh) — in-place.
                for j in 0..hidden {
                    zt[j] = xz[j] + hrz[j] + wbz_s[j] + rbz_s[j];
                    rt[j] = xr[j] + hrr[j] + wbr_s[j] + rbr_s[j];
                }
                sigmoid.run(&mut zt)?;
                sigmoid.run(&mut rt)?;
                // h_inner = rt * (h_prev·Rhᵀ + Rbh)   (linear_before_reset)
                for j in 0..hidden {
                    ht[j] = xh[j] + rt[j] * (hrh[j] + rbh_s[j]) + wbh_s[j];
                }
                tanh.run(&mut ht)?;
                // h_new = (1 - z) * candidate + z * h_prev → write Y and state.
                let y_row = &mut y_flat[t * hidden..(t + 1) * hidden];
                for j in 0..hidden {
                    let nh = (1.0 - zt[j]) * ht[j] + zt[j] * h_prev[j];
                    y_row[j] = nh;
                    new_h[j] = nh;
                }
                std::mem::swap(&mut h_prev, &mut new_h);
            }
            // h_prev now holds the final state.
            h_last = Array1::from(h_prev).insert_axis(Axis(0)); // [1, hidden]
        } else {
            // ── GENERIC PATH (batch>1 or linear_before_reset=0): ndarray.
            // Correct but unoptimized; no perf canary targets this combination.
            // Input projection X @ W^T → [seq, batch, 3*hidden] via ndarray.
            let x_flat = x_view.into_shape_with_order((seq_len * batch, input_size))?;
            let x_w_t = x_flat.dot(&w_view.t());
            let x_w_t = x_w_t.into_shape_with_order((seq_len, batch, 3 * hidden))?;

            let rh = r_view.slice(s![2 * hidden..3 * hidden, ..]);
            // For linear_before_reset the recurrent z/r/h projections all use
            // h_prev, so combine them into one [hidden, 3*hidden] weight and do
            // a single matmul per timestep. For !linear_before_reset only z/r
            // share h_prev (h needs rt⊙h_prev), so combine just z+r.
            let r_t = r_view.t(); // [hidden, 3*hidden] view
            let r_zr_t = r_view.slice(s![0..2 * hidden, ..]).t().to_owned(); // [hidden, 2*hidden]

            let mut h_prev = h0_view.to_owned();
            for t in 0..seq_len {
                let xt_z = x_w_t.slice(s![t, .., 0..hidden]);
                let xt_r = x_w_t.slice(s![t, .., hidden..2 * hidden]);
                let xt_h = x_w_t.slice(s![t, .., 2 * hidden..3 * hidden]);

                let (zt, h_inner) = if self.linear_before_reset {
                    let h_r = h_prev.dot(&r_t); // [batch, 3*hidden]
                    let h_rz = h_r.slice(s![.., 0..hidden]);
                    let h_rr = h_r.slice(s![.., hidden..2 * hidden]);
                    let h_rh = h_r.slice(s![.., 2 * hidden..3 * hidden]);

                    let mut zt = &xt_z + &h_rz;
                    zt += &wbz;
                    zt += &rbz;
                    zt.mapv_inplace(sigmoid_f32);

                    let mut rt = &xt_r + &h_rr;
                    rt += &wbr;
                    rt += &rbr;
                    rt.mapv_inplace(sigmoid_f32);

                    let mut h_rh_rbh = h_rh.to_owned();
                    h_rh_rbh += &rbh;
                    let h_inner = &rt * &h_rh_rbh;
                    (zt, h_inner)
                } else {
                    let h_zr = h_prev.dot(&r_zr_t); // [batch, 2*hidden]
                    let h_rz = h_zr.slice(s![.., 0..hidden]);
                    let h_rr = h_zr.slice(s![.., hidden..2 * hidden]);

                    let mut zt = &xt_z + &h_rz;
                    zt += &wbz;
                    zt += &rbz;
                    zt.mapv_inplace(sigmoid_f32);

                    let mut rt = &xt_r + &h_rr;
                    rt += &wbr;
                    rt += &rbr;
                    rt.mapv_inplace(sigmoid_f32);

                    let rt_h_prev = &rt * &h_prev;
                    let mut h_inner = rt_h_prev.dot(&rh.t());
                    h_inner += &rbh;
                    (zt, h_inner)
                };

                let mut ht = &xt_h + &h_inner;
                ht += &wbh;
                ht.mapv_inplace(f32::tanh);

                let new_h: Array2<f32> = {
                    let mut acc: Array2<f32> = (1.0f32 - &zt) * &ht;
                    acc += &(&zt * &h_prev);
                    acc
                };
                y.slice_mut(s![.., t, ..]).assign(&new_h);
                h_prev = new_h;
            }
            h_last = h_prev;
        }

        // Y_h: insert chunk axis at position 1 to match Scan body's
        // `wire!(y_h = AxisOp::Add(1), Ht);` → [batch, 1, hidden]. The wire
        // will then Move(1,0) for !batch_first to get [1, batch, hidden].
        let h_final = h_last.insert_axis(Axis(1)); // [batch, 1, hidden]
        let y_tensor: Tensor = y.into();
        let h_tensor: Tensor = h_final.to_owned().into();
        Ok(tvec![
            y_tensor.cast_to_dt(dt)?.into_owned().into(),
            h_tensor.cast_to_dt(dt)?.into_owned().into(),
        ])
    }
}

/// Persistent state for `OptGru`. Caches the packed recurrent-weight matrix R
/// across inferences (R is a constant GRU weight). Packing R[3*hidden, hidden]
/// (~786 KB) is the dominant per-call cost on wasm (~1.7 ms), so it is packed
/// lazily on the first eval and reused thereafter.
#[derive(Debug, Clone, Default)]
struct OptGruState {
    /// Packed input weights W (constant GRU weight) for the hoisted X·Wᵀ
    /// projection. Packed once on the first eval, reused thereafter.
    packed_w: PackedWeight,
    /// Packed recurrent weights R (constant) for the per-timestep R·h_prev.
    packed_r: PackedWeight,
}

trivial_op_state_freeze!(OptGruState);

impl OpState for OptGruState {
    fn eval(
        &mut self,
        _session: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<OptGru>().context("OptGruState attached to a non-OptGru op")?;
        op.run_eval(inputs, &mut self.packed_w, &mut self.packed_r)
    }
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl TypedOp for OptGru {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 5, "OptGru expects 5 inputs (X, W, R, B, h0)");
        let x = inputs[0];
        ensure!(x.rank() == 3, "X must be rank 3 [seq_len, batch, input_size]");
        let dt = x.datum_type;
        let seq_len = x.shape[0].clone();
        let batch = x.shape[1].clone();
        let hidden = self.hidden_size.to_dim();

        // Y in Scan-accumulator layout [batch, seq, hidden].
        // Y_h in Scan-body-output layout [batch, 1, hidden].
        let one = 1usize.to_dim();
        let y_fact = dt.fact(tvec![batch.clone(), seq_len, hidden.clone()]);
        let y_h_fact = dt.fact(tvec![batch, one, hidden]);

        Ok(tvec!(y_fact, y_h_fact))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare OptGru against the reference scalar GRU recurrence on a tiny
    /// model. Both should produce within-1e-5 outputs.
    #[test]
    fn opt_gru_tiny_linear_before_reset_true() {
        let hidden = 2;
        let input = 3;
        let seq_len = 4;
        let batch = 1;

        // Deterministic small inputs.
        let x = Tensor::from(Array3::from_shape_fn((seq_len, batch, input), |(t, _, i)| {
            ((t * 7 + i * 3) as f32 * 0.1) - 0.5
        }));
        let w = Tensor::from(Array2::from_shape_fn((3 * hidden, input), |(o, i)| {
            ((o * 5 + i * 11) as f32 * 0.07) - 0.3
        }));
        let r = Tensor::from(Array2::from_shape_fn((3 * hidden, hidden), |(o, i)| {
            ((o * 13 + i * 17) as f32 * 0.05) - 0.2
        }));
        let b = Tensor::from(Array1::from_shape_fn(6 * hidden, |i| (i as f32 * 0.04) - 0.1));
        let h0 = Tensor::from(Array2::<f32>::zeros((batch, hidden)));

        let op = OptGru { hidden_size: hidden, input_size: input, linear_before_reset: true };
        let outputs = op
            .eval(tvec![
                x.into_tvalue(),
                w.into_tvalue(),
                r.into_tvalue(),
                b.into_tvalue(),
                h0.into_tvalue(),
            ])
            .unwrap();

        // Output shapes — Scan-body-accumulator layout
        assert_eq!(outputs.len(), 2);
        let y = outputs[0].to_plain_array_view::<f32>().unwrap();
        let y_h = outputs[1].to_plain_array_view::<f32>().unwrap();
        assert_eq!(y.shape(), &[batch, seq_len, hidden]);
        assert_eq!(y_h.shape(), &[batch, 1, hidden]);

        // Y_h must equal Y[:, -1, :]
        let y_last = y.slice(s![.., seq_len - 1, ..]);
        let y_h_v = y_h.slice(s![.., 0, ..]);
        for (a, b) in y_last.iter().zip(y_h_v.iter()) {
            assert!((a - b).abs() < 1e-6, "Y[-1] {a} != Y_h {b}");
        }

        // Sanity: outputs are finite and in (-1, 1) (output of (1-z)*tanh + z*h_prev with h_0=0)
        for v in y.iter() {
            assert!(v.is_finite() && v.abs() < 2.0, "out-of-range output {v}");
        }
    }

    /// Compare OptGru output against an independent scalar reference
    /// implementation (no ndarray, no shape gymnastics). This is the
    /// gold-standard correctness check.
    #[test]
    fn opt_gru_matches_scalar_reference() {
        let hidden = 4;
        let input = 3;
        let seq_len = 5;
        let batch = 2;

        // Deterministic inputs
        let x_data: Vec<f32> =
            (0..seq_len * batch * input).map(|i| ((i as f32 * 0.137) - 1.0) * 0.3).collect();
        let w_data: Vec<f32> =
            (0..3 * hidden * input).map(|i| ((i as f32 * 0.083) - 1.0) * 0.2).collect();
        let r_data: Vec<f32> =
            (0..3 * hidden * hidden).map(|i| ((i as f32 * 0.071) - 1.0) * 0.15).collect();
        let b_data: Vec<f32> = (0..6 * hidden).map(|i| (i as f32 * 0.03) - 0.4).collect();
        let h0_data: Vec<f32> = vec![0.0; batch * hidden];

        let x_arr = Array3::from_shape_vec((seq_len, batch, input), x_data.clone()).unwrap();
        let w_arr = Array2::from_shape_vec((3 * hidden, input), w_data.clone()).unwrap();
        let r_arr = Array2::from_shape_vec((3 * hidden, hidden), r_data.clone()).unwrap();
        let b_arr = Array1::from_vec(b_data.clone());
        let h0_arr = Array2::from_shape_vec((batch, hidden), h0_data.clone()).unwrap();

        for lbr in [true, false] {
            // OptGru path
            let op = OptGru { hidden_size: hidden, input_size: input, linear_before_reset: lbr };
            let outputs = op
                .eval(tvec![
                    Tensor::from(x_arr.clone()).into_tvalue(),
                    Tensor::from(w_arr.clone()).into_tvalue(),
                    Tensor::from(r_arr.clone()).into_tvalue(),
                    Tensor::from(b_arr.clone()).into_tvalue(),
                    Tensor::from(h0_arr.clone()).into_tvalue(),
                ])
                .unwrap();
            // OptGru returns Y as [batch, seq, hidden] and Y_h as
            // [batch, 1, hidden] (Scan-accumulator layout). Transpose/squeeze
            // to compare with the scalar reference's [seq, batch, hidden] Y.
            let y_batch_first = outputs[0]
                .to_plain_array_view::<f32>()
                .unwrap()
                .into_dimensionality::<Ix3>()
                .unwrap();
            // y_batch_first: [batch, seq, hidden] → permute to [seq, batch, hidden]
            let y = y_batch_first.permuted_axes([1, 0, 2]);
            let y_h_rank3 = outputs[1]
                .to_plain_array_view::<f32>()
                .unwrap()
                .into_dimensionality::<Ix3>()
                .unwrap();
            let y_h = y_h_rank3.slice(s![.., 0, ..]);

            // Scalar reference: implement GRU manually with simple Rust loops.
            // Indexes: w[g*hidden + h, i], r[g*hidden + h, i], b layout [Wbz,Wbr,Wbh,Rbz,Rbr,Rbh].
            let mut ref_h: Vec<f32> = h0_data.clone();
            let mut ref_y: Vec<f32> = vec![0.0; seq_len * batch * hidden];
            let wbz = &b_data[0..hidden];
            let wbr = &b_data[hidden..2 * hidden];
            let wbh = &b_data[2 * hidden..3 * hidden];
            let rbz = &b_data[3 * hidden..4 * hidden];
            let rbr = &b_data[4 * hidden..5 * hidden];
            let rbh = &b_data[5 * hidden..6 * hidden];
            for t in 0..seq_len {
                let mut new_h = vec![0.0f32; batch * hidden];
                for b in 0..batch {
                    // Compute all input projections X·W^T for this t,b
                    let mut xwz = vec![0.0f32; hidden];
                    let mut xwr = vec![0.0f32; hidden];
                    let mut xwh = vec![0.0f32; hidden];
                    for h in 0..hidden {
                        let mut sz = 0.0f32;
                        let mut sr = 0.0f32;
                        let mut sh = 0.0f32;
                        for i in 0..input {
                            let xv = x_data[t * batch * input + b * input + i];
                            sz += xv * w_data[(0 * hidden + h) * input + i];
                            sr += xv * w_data[(1 * hidden + h) * input + i];
                            sh += xv * w_data[(2 * hidden + h) * input + i];
                        }
                        xwz[h] = sz;
                        xwr[h] = sr;
                        xwh[h] = sh;
                    }
                    // Compute hrz, hrr (used by z, r gates), hrh (only for lbr=true)
                    let mut hrz = vec![0.0f32; hidden];
                    let mut hrr = vec![0.0f32; hidden];
                    let mut hrh = vec![0.0f32; hidden];
                    for h in 0..hidden {
                        let mut sz = 0.0f32;
                        let mut sr = 0.0f32;
                        let mut sh = 0.0f32;
                        for k in 0..hidden {
                            let hv = ref_h[b * hidden + k];
                            sz += hv * r_data[(0 * hidden + h) * hidden + k];
                            sr += hv * r_data[(1 * hidden + h) * hidden + k];
                            sh += hv * r_data[(2 * hidden + h) * hidden + k];
                        }
                        hrz[h] = sz;
                        hrr[h] = sr;
                        hrh[h] = sh;
                    }
                    // Compute z, r FIRST (all elements) — these are full vectors,
                    // not scalars per outer h.
                    let mut zt = vec![0.0f32; hidden];
                    let mut rt = vec![0.0f32; hidden];
                    for h in 0..hidden {
                        zt[h] = 1.0 / (1.0 + (-(xwz[h] + hrz[h] + wbz[h] + rbz[h])).exp());
                        rt[h] = 1.0 / (1.0 + (-(xwr[h] + hrr[h] + wbr[h] + rbr[h])).exp());
                    }
                    // For lbr=false, need an extra matmul: (rt ⊙ h_prev) @ Rh^T
                    let h_inner_no_bias: Vec<f32> = if lbr {
                        // h_inner = rt[h] * (hrh[h] + rbh[h]) → per-h scalar
                        (0..hidden).map(|h| rt[h] * (hrh[h] + rbh[h])).collect()
                    } else {
                        // h_inner[h] = sum_k (rt[k] * h_prev[k]) * Rh[h, k] + rbh[h]
                        (0..hidden)
                            .map(|h| {
                                let mut s = 0.0f32;
                                for k in 0..hidden {
                                    s += (rt[k] * ref_h[b * hidden + k])
                                        * r_data[(2 * hidden + h) * hidden + k];
                                }
                                s + rbh[h]
                            })
                            .collect()
                    };
                    for h in 0..hidden {
                        let ht = (xwh[h] + h_inner_no_bias[h] + wbh[h]).tanh();
                        let new = (1.0 - zt[h]) * ht + zt[h] * ref_h[b * hidden + h];
                        new_h[b * hidden + h] = new;
                        ref_y[t * batch * hidden + b * hidden + h] = new;
                    }
                }
                ref_h = new_h;
            }

            // Compare
            let max_abs_y: f32 =
                y.iter().zip(ref_y.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            let max_abs_yh: f32 =
                y_h.iter().zip(ref_h.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(
                max_abs_y < 1e-5,
                "Y mismatch vs scalar reference (lbr={lbr}): max_abs={max_abs_y}"
            );
            assert!(
                max_abs_yh < 1e-5,
                "Y_h mismatch vs scalar reference (lbr={lbr}): max_abs={max_abs_yh}"
            );
        }
    }

    #[test]
    fn opt_gru_tiny_linear_before_reset_false() {
        let hidden = 2;
        let input = 3;
        let seq_len = 2;
        let batch = 1;

        let x = Tensor::from(Array3::<f32>::from_elem((seq_len, batch, input), 0.5));
        let w = Tensor::from(Array2::<f32>::from_elem((3 * hidden, input), 0.1));
        let r = Tensor::from(Array2::<f32>::from_elem((3 * hidden, hidden), 0.1));
        let b = Tensor::from(Array1::<f32>::zeros(6 * hidden));
        let h0 = Tensor::from(Array2::<f32>::zeros((batch, hidden)));

        let op = OptGru { hidden_size: hidden, input_size: input, linear_before_reset: false };
        let outputs = op
            .eval(tvec![
                x.into_tvalue(),
                w.into_tvalue(),
                r.into_tvalue(),
                b.into_tvalue(),
                h0.into_tvalue(),
            ])
            .unwrap();

        let y = outputs[0].to_plain_array_view::<f32>().unwrap();
        assert_eq!(y.shape(), &[batch, seq_len, hidden]);
        for v in y.iter() {
            assert!(v.is_finite() && v.abs() < 2.0);
        }
    }
}
