use crate::internal::*;
use tract_linalg::MmmDispatch;
use tract_linalg::mmm::{AsInputValue, FusedSpec, MMMInputValue, MatMatMul, Query};
use tract_linalg::pack::PackedFormat;
use tract_linalg::routines::Func;
use tract_ndarray::prelude::*;

/// The recurrent weight `R`, packed once and reused for every timestep of every
/// later call — see [`GruSeqState::packed_r`]. The `usize` is which of the
/// kernel's packings `R` was prepared in.
type PackedR = (Box<dyn MatMatMul>, usize, Box<dyn MMMInputValue>);

/// Whole-sequence GRU: the ONNX GRU with `linear_before_reset != 0`, run as one op
/// instead of a `Scan` that dispatches its body once per timestep.
///
/// Valid exactly where [`crate::ops::gru_cell::GruEpilogue`] is: sigmoid `f`, tanh
/// `g`, no peepholes, no extra cell state, no `sequence_lens`. `R` must be a
/// constant -- it is packed once and the packing is cached for the op's life.
///
/// The input-side product `X.Wt` does not depend on the recurrent state, so it is
/// taken once over the whole sequence as a single GEMM rather than once per step.
///
/// State: the op works both ways, as the `Scan` it replaces does. By default the
/// hidden state lives in the session and persists across calls -- `initial_h`
/// seeds the first call only -- so models keep their current behaviour bit for
/// bit. With `reset_every_turn` the state is re-seeded from `initial_h` on every
/// call instead, which is the ONNX contract and what a caller managing its own
/// state wants. The flag is named after and means the same as `Scan`'s.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct GruSeq {
    pub hidden: usize,
    pub has_bias: bool,
    /// -1 runs the sequence backwards (the `.back` side of a bidirectional GRU).
    pub chunk: isize,
    /// Fill the `Y` output with the whole sequence. False leaves it empty, for a
    /// caller that only reads `Y_h`.
    pub emit_y: bool,
    /// Re-seed the hidden state from `initial_h` on every call instead of
    /// carrying it in the session. Same meaning as `Scan::reset_every_turn`.
    pub reset_every_turn: bool,
}

#[derive(Default)]
struct GruSeqState {
    h: Option<Tensor>,
    /// R packed for the recurrent GEMM, built on the first call and reused for
    /// every timestep of every later call. Sound only because the wiring requires
    /// `R` to be a constant.
    packed_r: Option<PackedR>,
}

impl std::fmt::Debug for GruSeqState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("GruSeqState").field("h", &self.h).finish()
    }
}

impl Clone for GruSeqState {
    fn clone(&self) -> Self {
        // The packed weight is a cache; a clone rebuilds it on first use.
        GruSeqState { h: self.h.clone(), packed_r: None }
    }
}

impl Op for GruSeq {
    fn name(&self) -> StaticName {
        "GruSeq".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "hidden={} bias={} chunk={} reset_every_turn={} emit_y={}",
            self.hidden, self.has_bias, self.chunk, self.reset_every_turn, self.emit_y
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for GruSeq {
    not_out_of_plan!();

    /// Stateful even with `reset_every_turn`: the state also holds the packed `R`,
    /// which is a per-run cache rather than part of the recurrence.
    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<GruSeqState>::default()))
    }
}

impl OpState for GruSeqState {
    fn eval(
        &mut self,
        _ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<GruSeq>().context("wrong op")?;
        op.eval_with(&mut self.h, &mut self.packed_r, inputs)
    }

    /// GruSeq carries a single hidden state, not one row per stream, so it cannot
    /// be split across lanes -- same stance as the `Scan` it replaces.
    fn reset_lanes(&mut self, _lanes: &[LaneId]) -> TractResult<()> {
        bail!("GruSeq is not lane-aware: it carries a single hidden state")
    }
}

impl GruSeq {
    fn eval_with(
        &self,
        carry: &mut Option<Tensor>,
        packed_r: &mut Option<PackedR>,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (x, w, r) = (&inputs[0], &inputs[1], &inputs[2]);
        let b = if self.has_bias { Some(&inputs[3]) } else { None };
        let h0 = &inputs[inputs.len() - 1];
        let h = self.hidden;

        let x = x.to_plain_array_view::<f32>()?.into_dimensionality::<Ix3>()?; // [batch, T, in]
        let w = w.to_plain_array_view::<f32>()?.into_dimensionality::<Ix2>()?; // [3h, in]
        let (batch, t_len, in_size) = x.dim();

        let (wb, rb) = match b {
            Some(b) => {
                // B keeps its direction axis: [1, 6*hidden], unlike W and R.
                let v = b.to_plain_array_view::<f32>()?;
                let b = match v.ndim() {
                    2 => v.into_dimensionality::<Ix2>()?.index_axis_move(Axis(0), 0).to_owned(),
                    _ => v.into_dimensionality::<Ix1>()?.to_owned(),
                };
                (Some(b.slice(s![0..3 * h]).to_owned()), Some(b.slice(s![3 * h..6 * h]).to_owned()))
            }
            None => (None, None),
        };

        // One GEMM for the whole sequence, and the W-side bias folded in once here
        // rather than once per timestep. The step loop reads one timestep's rows at
        // a time, so the rows are ordered by timestep, not by batch element.
        let x_permuted = x.permuted_axes([1, 0, 2]);
        let x_by_step = x_permuted.as_standard_layout();
        let mut xw = x_by_step.to_shape((t_len * batch, in_size))?.dot(&w.t());
        if let Some(wb) = &wb {
            xw += &wb.view().insert_axis(Axis(0));
        }

        // Pack R once for the whole model's life, then run tract's own MMM per step
        // -- the same kernel the Scan body dispatches, so the arithmetic matches, but
        // without re-packing or re-dispatching a graph node each timestep.
        if packed_r.is_none() {
            // Computed transposed: R[3h, h] . h_prev[h, batch] -> [3h, batch].
            // With batch == 1 that is n == 1, which is how tract selects its
            // matrix-vector kernel -- the side that gets packed is R, once, and the
            // per-step vector is never packed. No extractor: R is packed here once,
            // so an extractor would re-run on every panel of every step.
            let query = Query {
                allow_extractor: false,
                ..Query::plain(f32::datum_type(), Some(3 * h), Some(h), Some(batch))
            };
            let (mmm, packing, _) = MmmDispatch::native()
                .pick(&query)
                .context("no matmul kernel for the recurrent product")?;
            let (pack_a, _) = &mmm.packings()[packing];
            let r_t = r.clone().into_tensor();
            let pa = pack_a.prepare_one(&r_t, 1, 0)?;
            *packed_r = Some((mmm, packing, pa));
        }
        let (mmm, packing, pa) = packed_r.as_ref().unwrap();
        let (_, pack_b) = &mmm.packings()[*packing];

        // With reset_every_turn the initializer wins every call; otherwise the
        // session's carry seeds every call but the first.
        let mut ht: Tensor = match carry.as_ref().filter(|_| !self.reset_every_turn) {
            Some(c) => squeeze_state(c, batch, h)?,
            None => squeeze_state(h0, batch, h)?,
        };

        let sigmoid = Func::Sigmoid.ew_f32()?;
        let tanh = Func::Tanh.ew_f32()?;

        // Everything the loop needs, allocated once -- including the packed form of
        // the per-step state. `prepare_one` would allocate a fresh panel buffer on
        // every timestep; the buffer's size depends only on `h` and `batch`, so it
        // is built here and refilled in place instead.
        let pf = pack_b
            .downcast_ref::<PackedFormat>()
            .context("recurrent product expects a plainly packed B side")?;
        let mut packed_ht = pf.new_packed_buffer(h, batch)?;
        // The recurrent product is stored straight into its [batch, 3*h] layout by
        // stride, so the step needs no transposing copy out of a [3*h, batch] temp.
        let mut rh = Tensor::zero::<f32>(&[batch, 3 * h])?;
        let mut h_next = Tensor::zero::<f32>(&[batch, h])?;
        let mut y = Array3::<f32>::zeros((batch, if self.emit_y { t_len } else { 0 }, h));

        for step in 0..t_len {
            let t = if self.chunk < 0 { t_len - 1 - step } else { step };

            // rh = h_prev . R^T (+ R-side bias), written into the same buffer each step.
            {
                // ht is [batch, h]; the product wants [h, batch], which the packer
                // reaches by stride, so the state never needs transposing into a
                // temporary.
                pf.repack_tensor_view(&mut packed_ht, &ht.view(), 1, 0)?;
                let pb = &packed_ht;
                unsafe {
                    let c = mmm.c_view(Some(1), Some(0)).wrap(&rh.view_mut());
                    mmm.run(
                        3 * h,
                        batch,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&**pa),
                                b: AsInputValue::Borrowed(pb),
                                packing: 0,
                            },
                            FusedSpec::Store(c),
                        ],
                    )?;
                }
                if let Some(rb) = &rb {
                    let rb = rb.as_slice().context("R-side bias not contiguous")?;
                    for row in rh.try_as_plain_mut()?.as_slice_mut::<f32>()?.chunks_mut(3 * h) {
                        for (o, b) in row.iter_mut().zip(rb) {
                            *o += b;
                        }
                    }
                }
            }

            let xh_row = &mut xw.as_slice_mut().context("xw not contiguous")?
                [t * batch * 3 * h..(t + 1) * batch * 3 * h];
            crate::ops::gru_cell::gru_cell_rows(
                h,
                batch,
                xh_row,
                rh.try_as_plain()?.as_slice::<f32>()?,
                ht.try_as_plain()?.as_slice::<f32>()?,
                h_next.try_as_plain_mut()?.as_slice_mut::<f32>()?,
                &*sigmoid,
                &*tanh,
            )?;
            std::mem::swap(&mut ht, &mut h_next);
            if self.emit_y {
                y.slice_mut(s![.., t, ..])
                    .assign(&ht.to_plain_array_view::<f32>()?.into_dimensionality::<Ix2>()?);
            }
        }

        *carry = if self.reset_every_turn { None } else { Some(ht.clone()) };
        let mut h_out = ht;
        h_out.insert_axis(1)?; // back to [batch, 1, hidden]
        Ok(tvec!(y.into_tensor().into(), h_out.into()))
    }
}

/// initial_h and the state slot are chunk-shaped [batch, 1, hidden].
fn squeeze_state(t: &Tensor, batch: usize, h: usize) -> TractResult<Tensor> {
    let mut t = t.clone().into_tensor();
    ensure!(
        t.len() == batch * h,
        "GruSeq state holds {} elements, expected batch {batch} x hidden {h}",
        t.len()
    );
    t.set_shape(&[batch, h])?;
    Ok(t)
}

impl TypedOp for GruSeq {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x = inputs[0];
        let batch = x.shape[0].clone();
        let t = x.shape[1].clone();
        let y_len = if self.emit_y { t } else { 0.to_dim() };
        Ok(tvec!(
            f32::fact([batch.clone(), y_len, self.hidden.to_dim()]),
            f32::fact([batch, 1.to_dim(), self.hidden.to_dim()])
        ))
    }
    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::gru_cell::GruEpilogue;

    /// The fused op must reproduce the per-timestep recurrence exactly: same gate
    /// maths, same order, so the only difference from a `Scan` is how often tract
    /// dispatches. Checked against a plain reference loop here; checked against the
    /// real `Scan` lowering end to end on GTCRN.
    fn reference(
        x: &Array3<f32>,
        w: &Array2<f32>,
        r: &Array2<f32>,
        b: Option<&Array1<f32>>,
        h0: &Array2<f32>,
        hidden: usize,
        backward: bool,
    ) -> (Array3<f32>, Array2<f32>) {
        let (batch, t_len, _) = x.dim();
        let mut ht = h0.clone();
        let mut y = Array3::<f32>::zeros((batch, t_len, hidden));
        for step in 0..t_len {
            let t = if backward { t_len - 1 - step } else { step };
            let mut xh = x.slice(s![.., t, ..]).to_owned().dot(&w.t());
            let mut rh = ht.dot(&r.t());
            if let Some(b) = b {
                xh += &b.slice(s![0..3 * hidden]).insert_axis(Axis(0));
                rh += &b.slice(s![3 * hidden..6 * hidden]).insert_axis(Axis(0));
            }
            let out = GruEpilogue { hidden }
                .eval(
                    &EvalContext::out_of_plan(),
                    tvec!(
                        xh.into_tensor().into(),
                        rh.into_tensor().into(),
                        ht.clone().into_tensor().into()
                    ),
                )
                .unwrap();
            ht = out[0]
                .to_plain_array_view::<f32>()
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .to_owned();
            y.slice_mut(s![.., t, ..]).assign(&ht);
        }
        (y, ht)
    }

    fn run_case(batch: usize, t_len: usize, backward: bool, bias: bool) {
        let (input, hidden) = (12usize, 16usize);
        let f = |n: usize, k: f32| Array1::from_iter((0..n).map(|i| ((i as f32) * k).sin() * 0.3));
        let x = f(batch * t_len * input, 0.7).into_shape_with_order((batch, t_len, input)).unwrap();
        let w = f(3 * hidden * input, 0.31).into_shape_with_order((3 * hidden, input)).unwrap();
        let r = f(3 * hidden * hidden, 0.17).into_shape_with_order((3 * hidden, hidden)).unwrap();
        let b = bias.then(|| f(6 * hidden, 0.11));
        let h0 = Array2::<f32>::zeros((batch, hidden));

        let (want_y, want_h) = reference(&x, &w, &r, b.as_ref(), &h0, hidden, backward);

        let op = GruSeq {
            hidden,
            has_bias: bias,
            chunk: if backward { -1 } else { 1 },
            reset_every_turn: false,
            emit_y: true,
        };
        let mut inputs: TVec<TValue> = tvec!(
            x.clone().into_tensor().into(),
            w.clone().into_tensor().into(),
            r.clone().into_tensor().into()
        );
        if let Some(b) = &b {
            inputs.push(b.clone().into_tensor().into());
        }
        inputs.push(h0.clone().into_tensor().into());
        let mut carry = None;
        let mut packed = None;
        let got = op.eval_with(&mut carry, &mut packed, inputs).unwrap();

        let got_y = got[0].clone().into_tensor();
        let got_h = got[1]
            .to_plain_array_view::<f32>()
            .unwrap()
            .into_dimensionality::<Ix3>()
            .unwrap()
            .index_axis_move(Axis(1), 0)
            .to_owned();

        // Tolerance, not equality: the reference GEMMs through `matrixmultiply` and
        // the op through tract's MMM. Their last bits differ on targets with no SIMD
        // mmm kernel, and the gap compounds over a 33-step recurrence. Bit-exactness
        // holds against the `Scan` this replaces, which shares the MMM path, and is
        // checked e2e.
        got_y.close_enough(&want_y.into_tensor(), Approximation::Approximate).unwrap_or_else(|e| {
            panic!("Y mismatch b={batch} t={t_len} backward={backward} bias={bias}: {e}")
        });
        got_h
            .into_tensor()
            .close_enough(&want_h.into_tensor(), Approximation::Approximate)
            .unwrap_or_else(|e| {
                panic!("Y_h mismatch b={batch} t={t_len} backward={backward} bias={bias}: {e}")
            });
    }

    #[test]
    fn matches_the_step_by_step_recurrence() {
        for &batch in &[1usize, 2, 3] {
            for &t in &[1usize, 2, 5, 33] {
                for &backward in &[false, true] {
                    for &bias in &[false, true] {
                        run_case(batch, t, backward, bias);
                    }
                }
            }
        }
    }

    /// The hidden state persists across calls, as the `Scan` it replaces does.
    #[test]
    fn carries_state_across_calls() {
        let op =
            GruSeq { hidden: 4, has_bias: false, chunk: 1, reset_every_turn: false, emit_y: true };
        let x = Array3::<f32>::from_elem((1, 3, 2), 0.5);
        let w = Array2::<f32>::from_elem((12, 2), 0.1);
        let r = Array2::<f32>::from_elem((12, 4), 0.1);
        let h0 = Array2::<f32>::zeros((1, 4));
        let mk = || -> TVec<TValue> {
            tvec!(
                x.clone().into_tensor().into(),
                w.clone().into_tensor().into(),
                r.clone().into_tensor().into(),
                h0.clone().into_tensor().into()
            )
        };
        let mut carry = None;
        let mut packed = None;
        let first = op.eval_with(&mut carry, &mut packed, mk()).unwrap();
        assert!(carry.is_some(), "state must be retained");
        let second = op.eval_with(&mut carry, &mut packed, mk()).unwrap();
        assert_ne!(
            first[1].to_plain_array_view::<f32>().unwrap(),
            second[1].to_plain_array_view::<f32>().unwrap(),
            "second call must continue from the carried state, not restart from initial_h"
        );
    }

    /// With reset_every_turn the initializer wins every call, so identical inputs
    /// give identical outputs -- the ONNX contract, and what a caller managing its
    /// own state across calls needs.
    #[test]
    fn reset_every_turn_restarts_from_initial_h() {
        let op =
            GruSeq { hidden: 4, has_bias: false, chunk: 1, reset_every_turn: true, emit_y: true };
        let x = Array3::<f32>::from_elem((1, 3, 2), 0.5);
        let w = Array2::<f32>::from_elem((12, 2), 0.1);
        let r = Array2::<f32>::from_elem((12, 4), 0.1);
        let h0 = Array2::<f32>::zeros((1, 4));
        let mk = || -> TVec<TValue> {
            tvec!(
                x.clone().into_tensor().into(),
                w.clone().into_tensor().into(),
                r.clone().into_tensor().into(),
                h0.clone().into_tensor().into()
            )
        };
        let mut carry = None;
        let mut packed = None;
        let first = op.eval_with(&mut carry, &mut packed, mk()).unwrap();
        assert!(carry.is_none(), "state must not be retained");
        let second = op.eval_with(&mut carry, &mut packed, mk()).unwrap();
        for slot in 0..2 {
            assert_eq!(
                first[slot].to_plain_array_view::<f32>().unwrap(),
                second[slot].to_plain_array_view::<f32>().unwrap(),
                "output {slot} must not drift between identical calls"
            );
        }
    }
}
