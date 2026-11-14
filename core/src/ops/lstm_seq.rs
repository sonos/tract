use crate::internal::*;
use tract_linalg::MmmDispatch;
use tract_linalg::mmm::{AsInputValue, FusedSpec, MMMInputValue, MatMatMul, Query};
use tract_linalg::pack::PackedFormat;
use tract_linalg::routines::Func;
use tract_ndarray::prelude::*;

/// The recurrent weight `R`, packed once and reused for every timestep of every
/// call. The `usize` is which of the kernel's packings `R` was prepared in.
type PackedR = (Box<dyn MatMatMul>, usize, Box<dyn MMMInputValue>);

/// Up to this hidden size the input-side product stays in the op: the graph
/// nodes projecting it out cost more to dispatch than the platform kernel saves.
const MAX_UNPROJECTED_HIDDEN: usize = 64;

/// Whole-sequence LSTM: the ONNX LSTM run as one op instead of a `Scan` that
/// dispatches its body once per timestep.
///
/// Valid exactly where [`crate::ops::lstm_cell::LstmEpilogue`] is: sigmoid `f`,
/// tanh `g` and `h`, no peepholes, no `sequence_lens`. `R` must be a constant.
/// Codegen packs it into the op when the batch size is known, so every state
/// spawned from the plan shares that packing; otherwise each state packs it on
/// its first call.
///
/// Inputs: `x` `[batch, T, in]`, `W` `[4*hidden, in]`, `R` `[4*hidden, hidden]`,
/// `B` `[1, 8*hidden]` when `has_bias`, then `initial_h` and `initial_c`, each
/// `[batch, 1, hidden]`. Gates are in ONNX order i, o, f, c. Outputs: `Y`
/// `[batch, T, hidden]` (empty unless `emit_y`), `Y_h` and `Y_c`
/// `[batch, 1, hidden]`.
///
/// The input-side product `X.Wt` does not depend on the recurrent state, so it
/// is taken once over the whole sequence rather than once per step. Above a
/// hidden size of 64, declutter moves it out of the op into the graph as an
/// `EinSum`, so it gets the platform's matmul kernel with `W` packed once, and
/// marks the op `projected`: input 0 is then `X.Wt`, `[batch, T, 4*hidden]`, and
/// there is no `W` input. Both bias halves stay in the op.
///
/// State: by default `h` and `c` live in the session and persist across calls
/// -- the initial states seed the first call only -- as with the `Scan` this
/// replaces. With `reset_every_turn` they are re-seeded on every call, which is
/// the ONNX contract.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LstmSeq {
    pub hidden: usize,
    pub has_bias: bool,
    /// -1 runs the sequence backwards (the `.back` side of a bidirectional LSTM).
    pub chunk: isize,
    /// Fill the `Y` output with the whole sequence. False leaves it empty, for a
    /// caller that only reads `Y_h` and `Y_c`.
    pub emit_y: bool,
    /// Re-seed `h` and `c` from the initial states on every call instead of
    /// carrying them in the session. Same meaning as `Scan::reset_every_turn`.
    pub reset_every_turn: bool,
    /// Input 0 is the input-side product `X.Wt` rather than `X`, and `W` is not
    /// an input. Set by declutter.
    pub projected: bool,
    /// `R` packed by codegen. `None` until then, or when the batch size is symbolic.
    pub packed_r: Option<PackedR>,
}

#[derive(Default)]
struct LstmSeqState {
    hc: Option<(Tensor, Tensor)>,
    /// R packed on the first call, for an op codegen did not pack it into. Sound
    /// only because the wiring requires `R` to be a constant.
    packed_r: Option<PackedR>,
}

impl std::fmt::Debug for LstmSeqState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("LstmSeqState").field("hc", &self.hc).finish()
    }
}

impl Clone for LstmSeqState {
    fn clone(&self) -> Self {
        LstmSeqState { hc: self.hc.clone(), packed_r: None }
    }
}

impl Op for LstmSeq {
    fn name(&self) -> StaticName {
        "LstmSeq".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "hidden={} bias={} chunk={} reset_every_turn={} emit_y={} projected={}",
            self.hidden,
            self.has_bias,
            self.chunk,
            self.reset_every_turn,
            self.emit_y,
            self.projected
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for LstmSeq {
    not_out_of_plan!();

    /// Stateful even with `reset_every_turn`: the state also holds the packed `R`.
    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<LstmSeqState>::default()))
    }
}

impl OpState for LstmSeqState {
    fn eval(
        &mut self,
        _ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<LstmSeq>().context("wrong op")?;
        op.eval_with(&mut self.hc, &mut self.packed_r, inputs)
    }

    fn reset_lanes(&mut self, _lanes: &[LaneId]) -> TractResult<()> {
        bail!("LstmSeq is not lane-aware: it carries a single hidden and cell state")
    }
}

impl LstmSeq {
    fn eval_with(
        &self,
        carry: &mut Option<(Tensor, Tensor)>,
        packed_r: &mut Option<PackedR>,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        // The f16 translator casts the inputs of ops it does not know to f16; the
        // recurrence runs in f32 whatever arrives.
        let inputs: TVec<std::borrow::Cow<Tensor>> =
            inputs.iter().map(|i| i.cast_to::<f32>()).collect::<TractResult<_>>()?;
        let r_slot = if self.projected { 1 } else { 2 };
        let r = &inputs[r_slot];
        let b = if self.has_bias { Some(&inputs[r_slot + 1]) } else { None };
        let h0 = &inputs[inputs.len() - 2];
        let c0 = &inputs[inputs.len() - 1];
        let h = self.hidden;
        let g = 4 * h;

        let x = inputs[0].to_plain_array_view::<f32>()?.into_dimensionality::<Ix3>()?;
        let (batch, t_len, in_size) = x.dim();

        // Both halves of B are added to every step's pre-activation, so they are
        // summed once, as the decomposed graph's constant-folded `bias_all` is.
        let bias = match b {
            Some(b) => {
                let b = b.to_plain_array_view::<f32>()?;
                let b = b.as_slice().context("LSTM bias not contiguous")?;
                ensure!(b.len() == 2 * g, "LSTM bias holds {} values, expected {}", b.len(), 2 * g);
                Some(b[..g].iter().zip(&b[g..]).map(|(w, r)| w + r).collect::<Vec<f32>>())
            }
            None => None,
        };

        // Rows ordered by timestep, so a step reads one contiguous block.
        let x_by_step = x.permuted_axes([1, 0, 2]);
        let x_by_step = x_by_step.as_standard_layout();
        let x_by_step = x_by_step.to_shape((t_len * batch, in_size))?;
        let mut xw = if self.projected {
            x_by_step.to_owned()
        } else {
            let w = inputs[1].to_plain_array_view::<f32>()?.into_dimensionality::<Ix2>()?;
            x_by_step.dot(&w.t())
        };

        if self.packed_r.is_none() && packed_r.is_none() {
            *packed_r = Some(pack_r(r, h, batch)?);
        }
        let (mmm, packing, pa) = self.packed_r.as_ref().or(packed_r.as_ref()).unwrap();
        let (_, pack_b) = &mmm.packings()[*packing];

        let (mut ht, mut ct) = match carry.as_ref().filter(|_| !self.reset_every_turn) {
            Some((hc, cc)) => (squeeze_state(hc, batch, h)?, squeeze_state(cc, batch, h)?),
            None => (squeeze_state(h0, batch, h)?, squeeze_state(c0, batch, h)?),
        };

        let sigmoid = Func::Sigmoid.ew_f32()?;
        let tanh = Func::Tanh.ew_f32()?;

        let pf = pack_b
            .downcast_ref::<PackedFormat>()
            .context("recurrent product expects a plainly packed B side")?;
        let mut packed_ht = pf.new_packed_buffer(h, batch)?;
        let mut rh = Tensor::zero::<f32>(&[batch, g])?;
        let mut h_next = Tensor::zero::<f32>(&[batch, h])?;
        let mut c_next = Tensor::zero::<f32>(&[batch, h])?;
        let mut y = Array3::<f32>::zeros((batch, if self.emit_y { t_len } else { 0 }, h));

        for step in 0..t_len {
            let t = if self.chunk < 0 { t_len - 1 - step } else { step };

            pf.repack_tensor_view(&mut packed_ht, &ht.view(), 1, 0)?;
            unsafe {
                let c = mmm.c_view(Some(1), Some(0)).wrap(&rh.view_mut());
                mmm.run(
                    g,
                    batch,
                    &[
                        FusedSpec::AddMatMul {
                            a: AsInputValue::Borrowed(&**pa),
                            b: AsInputValue::Borrowed(&packed_ht),
                            packing: 0,
                        },
                        FusedSpec::Store(c),
                    ],
                )?;
            }

            let pre = &mut xw.as_slice_mut().context("xw not contiguous")?
                [t * batch * g..(t + 1) * batch * g];
            let rh_s = rh.try_as_plain_ram()?.as_slice::<f32>()?;
            for (pre_row, rh_row) in pre.chunks_mut(g).zip(rh_s.chunks(g)) {
                for (p, r) in pre_row.iter_mut().zip(rh_row) {
                    *p += r;
                }
                if let Some(bias) = &bias {
                    for (p, b) in pre_row.iter_mut().zip(bias) {
                        *p += b;
                    }
                }
            }
            crate::ops::lstm_cell::lstm_cell_rows(
                h,
                batch,
                pre,
                ct.try_as_plain_ram()?.as_slice::<f32>()?,
                h_next.try_as_plain_ram_mut()?.as_slice_mut::<f32>()?,
                c_next.try_as_plain_ram_mut()?.as_slice_mut::<f32>()?,
                &*sigmoid,
                &*tanh,
            )?;
            std::mem::swap(&mut ht, &mut h_next);
            std::mem::swap(&mut ct, &mut c_next);
            if self.emit_y {
                y.slice_mut(s![.., t, ..])
                    .assign(&ht.to_plain_array_view::<f32>()?.into_dimensionality::<Ix2>()?);
            }
        }

        *carry = if self.reset_every_turn { None } else { Some((ht.clone(), ct.clone())) };
        let mut h_out = ht;
        h_out.insert_axis(1)?;
        let mut c_out = ct;
        c_out.insert_axis(1)?;
        Ok(tvec!(y.into_tensor().into(), h_out.into(), c_out.into()))
    }
}

/// Packs `R` for the recurrent product, computed transposed: R[4h, h] . h_prev[h, batch]
/// -> [4h, batch], so batch 1 selects tract's matrix-vector kernel. No extractor: R is
/// packed once, so an extractor would re-run on every panel of every step.
fn pack_r(r: &Tensor, h: usize, batch: usize) -> TractResult<PackedR> {
    let query = Query {
        allow_extractor: false,
        ..Query::plain(f32::datum_type(), Some(4 * h), Some(h), Some(batch))
    };
    let (mmm, packing, _) =
        MmmDispatch::native().pick(&query).context("no matmul kernel for the recurrent product")?;
    let (pack_a, _) = &mmm.packings()[packing];
    let pa = pack_a.prepare_one(r, 1, 0)?;
    Ok((mmm, packing, pa))
}

/// The initial states and the state slots are chunk-shaped [batch, 1, hidden].
fn squeeze_state(t: &Tensor, batch: usize, h: usize) -> TractResult<Tensor> {
    let mut t = t.clone().into_tensor();
    ensure!(
        t.len() == batch * h,
        "LstmSeq state holds {} elements, expected batch {batch} x hidden {h}",
        t.len()
    );
    t.set_shape(&[batch, h])?;
    Ok(t)
}

impl TypedOp for LstmSeq {
    /// Moves the input-side product into the graph above a hidden size of 64, so it
    /// runs on the platform's matmul kernel with `W` packed once.
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.projected || self.hidden <= MAX_UNPROJECTED_HIDDEN {
            return Ok(None);
        }
        let mut patch = TypedModelPatch::default();
        let taps = patch.taps(model, &node.inputs)?;
        let name = &node.name;
        let xw = patch.wire_node(
            format!("{name}.xw"),
            crate::ops::einsum::EinSum {
                axes: "btk,gk->btg".parse()?,
                operating_dt: f32::datum_type(),
                q_params: None,
            },
            &[taps[0], taps[1]],
        )?[0];
        let mut inputs = tvec!(xw);
        inputs.extend(taps[2..].iter().copied());
        let op = LstmSeq { projected: true, ..self.clone() };
        let outputs = patch.wire_node(name, op, &inputs)?;
        for (ix, o) in outputs.iter().enumerate() {
            patch.shunt_outside(model, OutletId::new(node.id, ix), *o)?;
        }
        Ok(Some(patch))
    }

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x = inputs[0];
        let batch = x.shape[0].clone();
        let t = x.shape[1].clone();
        let y_len = if self.emit_y { t } else { 0.to_dim() };
        let state = f32::fact([batch.clone(), 1.to_dim(), self.hidden.to_dim()]);
        Ok(tvec!(f32::fact([batch, y_len, self.hidden.to_dim()]), state.clone(), state))
    }

    /// Packs a constant `R` into the op, so states spawned from the plan do not each
    /// pack it on their first call. Needs a concrete batch size: the kernel is picked
    /// for it.
    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.packed_r.is_some() {
            return Ok(None);
        }
        let r_slot = if self.projected { 1 } else { 2 };
        let Some(r) = model.outlet_fact(node.inputs[r_slot])?.konst.clone() else {
            return Ok(None);
        };
        let Ok(batch) = model.outlet_fact(node.inputs[0])?.shape[0].to_usize() else {
            return Ok(None);
        };
        let r = r.cast_to::<f32>()?;
        let op = LstmSeq { packed_r: Some(pack_r(&r, self.hidden, batch)?), ..self.clone() };
        Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, op)?))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::lstm_cell::LstmEpilogue;

    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &Array3<f32>,
        w: &Array2<f32>,
        r: &Array2<f32>,
        b: Option<&Array1<f32>>,
        h0: &Array2<f32>,
        c0: &Array2<f32>,
        hidden: usize,
        backward: bool,
    ) -> (Array3<f32>, Array2<f32>, Array2<f32>) {
        let (batch, t_len, _) = x.dim();
        let (mut ht, mut ct) = (h0.clone(), c0.clone());
        let mut y = Array3::<f32>::zeros((batch, t_len, hidden));
        for step in 0..t_len {
            let t = if backward { t_len - 1 - step } else { step };
            let mut pre = x.slice(s![.., t, ..]).to_owned().dot(&w.t()) + ht.dot(&r.t());
            if let Some(b) = b {
                pre += &(&b.slice(s![0..4 * hidden]) + &b.slice(s![4 * hidden..8 * hidden]))
                    .insert_axis(Axis(0));
            }
            let out = LstmEpilogue { hidden }
                .eval(
                    &EvalContext::out_of_plan(),
                    tvec!(pre.into_tensor().into(), ct.clone().into_tensor().into()),
                )
                .unwrap();
            let as2 = |v: &TValue| {
                v.to_plain_array_view::<f32>()
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .to_owned()
            };
            ht = as2(&out[0]);
            ct = as2(&out[1]);
            y.slice_mut(s![.., t, ..]).assign(&ht);
        }
        (y, ht, ct)
    }

    fn f(n: usize, k: f32) -> Array1<f32> {
        Array1::from_iter((0..n).map(|i| ((i as f32) * k).sin() * 0.3))
    }

    fn state_2d(v: &TValue) -> Array2<f32> {
        v.to_plain_array_view::<f32>()
            .unwrap()
            .into_dimensionality::<Ix3>()
            .unwrap()
            .index_axis_move(Axis(1), 0)
            .to_owned()
    }

    fn op(hidden: usize, bias: bool, backward: bool, reset_every_turn: bool) -> LstmSeq {
        LstmSeq {
            hidden,
            has_bias: bias,
            chunk: if backward { -1 } else { 1 },
            reset_every_turn,
            emit_y: true,
            projected: false,
            packed_r: None,
        }
    }

    fn run_case(batch: usize, t_len: usize, backward: bool, bias: bool) {
        let (input, hidden) = (12usize, 16usize);
        let x = f(batch * t_len * input, 0.7).into_shape_with_order((batch, t_len, input)).unwrap();
        let w = f(4 * hidden * input, 0.31).into_shape_with_order((4 * hidden, input)).unwrap();
        let r = f(4 * hidden * hidden, 0.17).into_shape_with_order((4 * hidden, hidden)).unwrap();
        let b = bias.then(|| f(8 * hidden, 0.11));
        let h0 = f(batch * hidden, 0.23).into_shape_with_order((batch, hidden)).unwrap();
        let c0 = f(batch * hidden, 0.41).into_shape_with_order((batch, hidden)).unwrap();

        let (want_y, want_h, want_c) =
            reference(&x, &w, &r, b.as_ref(), &h0, &c0, hidden, backward);

        let mut inputs: TVec<TValue> =
            tvec!(x.into_tensor().into(), w.into_tensor().into(), r.into_tensor().into());
        if let Some(b) = b {
            inputs.push(b.into_tensor().into());
        }
        inputs.push(h0.insert_axis(Axis(1)).into_tensor().into());
        inputs.push(c0.insert_axis(Axis(1)).into_tensor().into());
        let got =
            op(hidden, bias, backward, false).eval_with(&mut None, &mut None, inputs).unwrap();

        let ctx = format!("b={batch} t={t_len} backward={backward} bias={bias}");
        let check = |got: Tensor, want: Tensor, what: &str| {
            got.close_enough(&want, Approximation::Approximate)
                .unwrap_or_else(|e| panic!("{what} mismatch {ctx}: {e}"))
        };
        check(got[0].clone().into_tensor(), want_y.into_tensor(), "Y");
        check(state_2d(&got[1]).into_tensor(), want_h.into_tensor(), "Y_h");
        check(state_2d(&got[2]).into_tensor(), want_c.into_tensor(), "Y_c");
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

    fn small_inputs() -> TVec<TValue> {
        tvec!(
            Array3::<f32>::from_elem((1, 3, 2), 0.5).into_tensor().into(),
            Array2::<f32>::from_elem((16, 2), 0.1).into_tensor().into(),
            Array2::<f32>::from_elem((16, 4), 0.1).into_tensor().into(),
            Array3::<f32>::zeros((1, 1, 4)).into_tensor().into(),
            Array3::<f32>::zeros((1, 1, 4)).into_tensor().into()
        )
    }

    /// `h` and `c` persist across calls, as with the `Scan` this replaces.
    #[test]
    fn carries_state_across_calls() {
        let op = op(4, false, false, false);
        let (mut carry, mut packed) = (None, None);
        let first = op.eval_with(&mut carry, &mut packed, small_inputs()).unwrap();
        assert!(carry.is_some(), "state must be retained");
        let second = op.eval_with(&mut carry, &mut packed, small_inputs()).unwrap();
        for slot in 1..3 {
            assert_ne!(
                first[slot].to_plain_array_view::<f32>().unwrap(),
                second[slot].to_plain_array_view::<f32>().unwrap(),
                "output {slot}: the second call must continue from the carried state"
            );
        }
    }

    /// With `reset_every_turn` identical inputs give identical outputs.
    #[test]
    fn reset_every_turn_restarts_from_initial_states() {
        let op = op(4, false, false, true);
        let (mut carry, mut packed) = (None, None);
        let first = op.eval_with(&mut carry, &mut packed, small_inputs()).unwrap();
        assert!(carry.is_none(), "state must not be retained");
        let second = op.eval_with(&mut carry, &mut packed, small_inputs()).unwrap();
        assert_eq!(first, second);
    }

    /// Inputs cast to f16, as the f16 translator does for ops it does not know,
    /// run in f32 and land within f16 rounding of the f32 run.
    #[test]
    fn f16_inputs_run_in_f32() {
        let op = op(4, false, false, true);
        let want = op.eval_with(&mut None, &mut None, small_inputs()).unwrap();
        let halved: TVec<TValue> = small_inputs()
            .iter()
            .map(|i| i.cast_to::<f16>().unwrap().into_owned().into())
            .collect();
        let got = op.eval_with(&mut None, &mut None, halved).unwrap();
        for (g, w) in got.iter().zip(&want) {
            g.close_enough(w, Approximation::SuperApproximate).unwrap();
        }
    }

    /// Optimized through declutter and codegen, forwards and backwards, with and
    /// without bias, with a known or symbolic batch: projected iff hidden > 64, `R`
    /// packed by codegen iff the batch is known, and still the step-by-step result.
    #[test]
    fn optimized_matches_the_recurrence() {
        let (batch, input, t_len) = (2usize, 12usize, 6usize);
        for &hidden in &[16usize, 80] {
            for case in 0..8 {
                let (backward, bias, symbolic) = (case & 1 != 0, case & 2 != 0, case & 4 != 0);
                let x = f(batch * t_len * input, 0.7)
                    .into_shape_with_order((batch, t_len, input))
                    .unwrap();
                let w =
                    f(4 * hidden * input, 0.31).into_shape_with_order((4 * hidden, input)).unwrap();
                let r = f(4 * hidden * hidden, 0.17)
                    .into_shape_with_order((4 * hidden, hidden))
                    .unwrap();
                let b = bias.then(|| f(8 * hidden, 0.11));
                let h0 = f(batch * hidden, 0.23).into_shape_with_order((batch, hidden)).unwrap();
                let c0 = f(batch * hidden, 0.41).into_shape_with_order((batch, hidden)).unwrap();
                let (want_y, want_h, want_c) =
                    reference(&x, &w, &r, b.as_ref(), &h0, &c0, hidden, backward);

                let mut model = TypedModel::default();
                let batch_dim =
                    if symbolic { model.symbols.sym("B").to_dim() } else { batch.to_dim() };
                let source = model
                    .add_source("x", f32::fact(&[batch_dim, t_len.to_dim(), input.to_dim()]))
                    .unwrap();
                let mut inputs = tvec!(
                    source,
                    model.add_const("w", w.into_tensor()).unwrap(),
                    model.add_const("r", r.into_tensor()).unwrap()
                );
                if let Some(b) = b {
                    inputs.push(model.add_const("b", b.into_tensor()).unwrap());
                }
                inputs.push(model.add_const("h0", h0.insert_axis(Axis(1)).into_tensor()).unwrap());
                inputs.push(model.add_const("c0", c0.insert_axis(Axis(1)).into_tensor()).unwrap());
                let outputs =
                    model.wire_node("lstm", op(hidden, bias, backward, true), &inputs).unwrap();
                model.select_output_outlets(&outputs).unwrap();
                let model = model.into_optimized().unwrap();
                let fused = model.nodes.iter().find_map(|n| n.op_as::<LstmSeq>()).unwrap();
                assert_eq!(fused.projected, hidden > MAX_UNPROJECTED_HIDDEN);
                assert_eq!(fused.packed_r.is_some(), !symbolic);

                let got =
                    model.into_runnable().unwrap().run(tvec!(x.into_tensor().into())).unwrap();
                let ok = |got: Tensor, want: Tensor| {
                    got.close_enough(&want, Approximation::Approximate).unwrap()
                };
                ok(got[0].clone().into_tensor(), want_y.into_tensor());
                ok(state_2d(&got[1]).into_tensor(), want_h.into_tensor());
                ok(state_2d(&got[2]).into_tensor(), want_c.into_tensor());
            }
        }
    }
}
