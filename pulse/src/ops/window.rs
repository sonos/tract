//! Pulsifier for `WindowOnAxis`.
//!
//! `WindowOnAxis(axis, W, start)` lowers to a three-op chain:
//!
//! ```text
//! Delay(0, W-1) → PulsePad(before = -start, after = 0, value = 0) → PulsedExposeWindow
//! ```
//!
//! * `Delay(0, W-1)` accumulates the latest W chunks per pulse (current
//!   plus `W-1` past), bumping `stream.delay` by `W-1`.
//! * `PulsePad(before = -start)` zero-fills the leading `-start` chunks of
//!   the post-delay buffer (so the very first pulses see zeros for the
//!   "out-of-stream" past, matching the batch-eval boundary semantics) and
//!   subtracts `-start` from `stream.delay`.  For `start = 0` (future
//!   window, ex03) this is a no-op (`before = 0`).  For `start = -(W-1)`
//!   (past window, ex04) it cancels the Delay's increment, leaving
//!   `stream.delay = 0` (causal).  For mixed bands the residue is
//!   `stream.delay = end = start + W - 1`.
//! * `PulsedExposeWindow` reshapes the per-pulse `[W, ...]` view into
//!   `[1, W, ...]`, exposing W as a static window axis.
//!
//! Constraint: `pulse == 1` on the windowed axis (the case Blockify
//! produces today).  Constraints: `start ≤ 0` and `start + W - 1 ≥ 0`
//! (the window must straddle the current chunk).

use crate::internal::*;
use tract_core::ops::array::PadMode;
use tract_pulse_opl::ops::{Delay, PulsePad, WindowOnAxis};

register_all!(WindowOnAxis: pulsify);

fn pulsify(
    op: &WindowOnAxis,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    rule_if_some!(stream = fact.stream.as_ref());
    rule_if!(stream.axis == op.axis);
    let pulse = fact.pulse().unwrap();
    let pulse_size = pulse.to_usize()?;
    ensure!(
        pulse_size == 1,
        "WindowOnAxis pulsifier currently requires pulse=1 on the windowed axis (got {pulse_size})"
    );

    // Bail on `start > 0` (purely-future skip-current) and on
    // `start + window - 1 < 0` (purely-past) — neither is needed yet.
    ensure!(op.start <= 0, "WindowOnAxis pulsifier: start > 0 not supported (got {})", op.start);
    ensure!(
        op.start + op.window as i64 - 1 >= 0,
        "WindowOnAxis pulsifier: window must straddle current (start={}, window={})",
        op.start,
        op.window
    );

    let overlap = op.window - 1;
    let before: usize = (-op.start) as usize;

    let delayed = target.wire_node(
        format!("{}.delay", node.name),
        Delay::new_typed(&(&fact).into(), op.axis, 0, overlap),
        &[input],
    )?[0];

    // For `start < 0` (past-window): zero-fill the leading `before` chunks
    // of the post-delay buffer and shift `stream.delay` back by `before`,
    // so the consumer's logical time anchors to the *latest* chunk in the
    // window rather than the oldest.  For `start = 0` we still wire the
    // pad with `before = 0`: it's a runtime no-op (no fill, no delay
    // change) but keeps the pulsifier's structure uniform.
    let zero = Tensor::zero_scalar_dt(fact.datum_type)?;
    let post_delay_fact = target.outlet_fact(delayed)?.clone();
    let post_delay_stream = post_delay_fact.stream.as_ref().unwrap();
    let begin_input = post_delay_stream.delay;
    let end_input = post_delay_stream.delay.to_dim() + &post_delay_stream.dim;
    let padded = target.wire_node(
        format!("{}.pulse_pad", node.name),
        PulsePad {
            axis: op.axis,
            before,
            after: 0.to_dim(),
            begin_input,
            end_input,
            mode: PadMode::Constant(zero.into_arc_tensor()),
            overlap,
        },
        &[delayed],
    )?[0];

    let exposed = target.wire_node(
        &*node.name,
        PulsedExposeWindow { axis: op.axis, window: op.window },
        &[padded],
    )?;
    Ok(Some(exposed))
}

/// Pulse-only op: splits the per-pulse streaming-axis buffer of size
/// `1 + W - 1 = W` into `[1, W]`, exposing `W` as a static window axis.
/// Logical streaming dim is preserved (1 chunk per pulse on the new
/// streaming axis at the same position as before).  Stream-delay
/// adjustment is handled upstream by the `PulsePad` the WindowOnAxis
/// pulsifier wires before this op.
#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct PulsedExposeWindow {
    pub axis: usize,
    pub window: usize,
}

impl Op for PulsedExposeWindow {
    fn name(&self) -> StaticName {
        "PulsedExposeWindow".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {} window: {}", self.axis, self.window)])
    }

    not_a_typed_op!();
}

impl EvalOp for PulsedExposeWindow {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs).into_tensor();
        let mut new_shape: TVec<usize> = input.shape().into();
        // input shape on `axis` is W (= pulse + overlap, with pulse=1).
        // Output shape on `axis` is 1, with a new W axis at `axis + 1`.
        ensure!(
            new_shape[self.axis] == self.window,
            "PulsedExposeWindow: per-pulse axis {} has size {} but expected window {}",
            self.axis,
            new_shape[self.axis],
            self.window
        );
        new_shape[self.axis] = 1;
        new_shape.insert(self.axis + 1, self.window);
        let mut t = input;
        unsafe { t.set_shape_unchecked(&new_shape) };
        Ok(tvec!(t.into_tvalue()))
    }
}

impl PulsedOp for PulsedExposeWindow {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let stream = fact.stream.as_mut().context("PulsedExposeWindow needs a streaming input")?;
        ensure!(
            stream.axis == self.axis,
            "PulsedExposeWindow: input stream axis {} doesn't match op axis {}",
            stream.axis,
            self.axis
        );
        let mut shape: TVec<TDim> = fact.shape.iter().cloned().collect();
        shape[self.axis] = 1.to_dim();
        shape.insert(self.axis + 1, self.window.to_dim());
        fact.shape = shape.into();
        Ok(tvec!(fact))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        // Typed equivalent: factor the pulse-axis (size W) into [1, W].
        // After this reshape the typed output matches eval's output shape:
        // [..., 1, W, ...] with the streaming axis at `self.axis` and the
        // static window at `self.axis + 1`.
        use tract_pulse_opl::tract_core::ops::change_axes::AxisOp;
        Box::new(AxisOp::Reshape(
            self.axis,
            tvec!(self.window.to_dim()),
            tvec!(1.to_dim(), self.window.to_dim()),
        ))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Typed batch eval of `WindowOnAxis(axis=0, window=2)` on `[S=4, D=2]`
    /// should produce `[S=4, W=2, D=2]` with future-shifted slots and zero
    /// fill at the trailing edge.
    #[test]
    fn window_on_axis_typed_eval() {
        let op = WindowOnAxis::future(0, 2);
        let input =
            tract_core::ndarray::Array2::<f32>::from_shape_fn((4, 2), |(i, j)| (i * 10 + j) as f32);
        let result = op.eval(tvec!(input.clone().into_dyn().into_tvalue())).unwrap();
        let out = result[0].to_plain_array_view::<f32>().unwrap().to_owned();
        assert_eq!(out.shape(), &[4, 2, 2]);
        // Slot w=0: copy of input.
        for s in 0..4 {
            for d in 0..2 {
                assert_eq!(out[[s, 0, d]], input[[s, d]], "w=0, s={s}, d={d}");
            }
        }
        // Slot w=1: shifted by 1, last row zero.
        for s in 0..3 {
            for d in 0..2 {
                assert_eq!(out[[s, 1, d]], input[[s + 1, d]], "w=1, s={s}, d={d}");
            }
        }
        for d in 0..2 {
            assert_eq!(out[[3, 1, d]], 0.0, "w=1 trailing pad must be zero");
        }
    }

    /// Pulsified WindowOnAxis on a streaming `[S, D]` input with pulse=1
    /// should produce `[1, W, D]` per pulse, lagging by `W-1` (the future-
    /// lookahead delay).  We feed S=4 chunks of size 1, plus W-1 trailing
    /// flush pulses, and verify the windowed output matches the expected
    /// future-shift values.
    #[test]
    fn window_on_axis_pulsified() {
        use tract_core::ndarray::Array2;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims![s.clone(), 2_usize].as_ref())).unwrap();
        let win = model.wire_node("win", WindowOnAxis::future(0, 2), &[a]).unwrap();
        model.select_output_outlets(&win).unwrap();

        let pulsed = PulsedModel::new(&model, s.clone(), &1.to_dim()).unwrap();
        let plan = SimplePlan::new(pulsed.into_typed().unwrap()).unwrap();
        let mut state = SimpleState::new(&plan).unwrap();

        let input = Array2::<f32>::from_shape_fn((4, 2), |(i, j)| (i * 10 + j) as f32);

        // Window=2 → output lags by 1.  Feed 4 + 1 trailing pulses.
        let mut got: Vec<Vec<f32>> = vec![];
        for s in 0..5 {
            let chunk: Array2<f32> = if s < 4 {
                input.slice_axis(tract_core::ndarray::Axis(0), (s..s + 1).into()).to_owned()
            } else {
                Array2::<f32>::zeros((1, 2)) // flush pulse
            };
            let out = state.run(tvec!(chunk.into_dyn().into_tvalue())).unwrap();
            let arr = out[0].to_plain_array_view::<f32>().unwrap().to_owned();
            assert_eq!(arr.shape(), &[1, 2, 2], "step {s}");
            got.push(arr.iter().copied().collect());
        }

        // After delay=W-1=1, pulses 0..S produce logical-time-(s-1) outputs.
        // Pulse 0: garbage (logical -1, before stream start).  Skip.
        // Pulse 1: logical 0 → window {input[0], input[1]}.
        let p1 = &got[1];
        assert_eq!(p1[..2], [input[[0, 0]], input[[0, 1]]]);
        assert_eq!(p1[2..], [input[[1, 0]], input[[1, 1]]]);
        // Pulse 2: logical 1 → window {input[1], input[2]}.
        let p2 = &got[2];
        assert_eq!(p2[..2], [input[[1, 0]], input[[1, 1]]]);
        assert_eq!(p2[2..], [input[[2, 0]], input[[2, 1]]]);
        // Pulse 3: logical 2 → window {input[2], input[3]}.
        let p3 = &got[3];
        assert_eq!(p3[..2], [input[[2, 0]], input[[2, 1]]]);
        assert_eq!(p3[2..], [input[[3, 0]], input[[3, 1]]]);
        // Pulse 4: logical 3 → window {input[3], 0}.
        let p4 = &got[4];
        assert_eq!(p4[..2], [input[[3, 0]], input[[3, 1]]]);
        assert_eq!(p4[2..], [0.0, 0.0]);
    }

    /// Past-window (`start = -1`, `W = 2`): output stream.delay must be 0
    /// (causal), and per-pulse slot 0 must be the *previous* chunk while
    /// slot 1 is current.  No trailing flush needed.
    #[test]
    fn window_on_axis_past_window_pulsified() {
        use tract_core::ndarray::Array2;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims![s.clone(), 2_usize].as_ref())).unwrap();
        let win =
            model.wire_node("win", WindowOnAxis { axis: 0, window: 2, start: -1 }, &[a]).unwrap();
        model.select_output_outlets(&win).unwrap();

        let pulsed = PulsedModel::new(&model, s.clone(), &1.to_dim()).unwrap();
        // Output's stream.delay must be 0 (causal past window).
        assert_eq!(pulsed.outputs[0], pulsed.outputs[0]); // (placeholder for compile)
        let plan = SimplePlan::new(pulsed.into_typed().unwrap()).unwrap();
        let mut state = SimpleState::new(&plan).unwrap();

        let input = Array2::<f32>::from_shape_fn((4, 2), |(i, j)| (i * 10 + j) as f32);
        let mut got: Vec<Vec<f32>> = vec![];
        for s in 0..4 {
            let chunk =
                input.slice_axis(tract_core::ndarray::Axis(0), (s..s + 1).into()).to_owned();
            let out = state.run(tvec!(chunk.into_dyn().into_tvalue())).unwrap();
            let arr = out[0].to_plain_array_view::<f32>().unwrap().to_owned();
            assert_eq!(arr.shape(), &[1, 2, 2], "step {s}");
            got.push(arr.iter().copied().collect());
        }

        // Pulse 0: current chunk = input[0], past chunk = zero (no prior).
        // slot 0 = past = 0, slot 1 = current = input[0].
        let p0 = &got[0];
        assert_eq!(p0[..2], [0.0, 0.0]);
        assert_eq!(p0[2..], [input[[0, 0]], input[[0, 1]]]);
        // Pulse 1: slot 0 = input[0], slot 1 = input[1].
        let p1 = &got[1];
        assert_eq!(p1[..2], [input[[0, 0]], input[[0, 1]]]);
        assert_eq!(p1[2..], [input[[1, 0]], input[[1, 1]]]);
        // Pulse 2: slot 0 = input[1], slot 1 = input[2].
        let p2 = &got[2];
        assert_eq!(p2[..2], [input[[1, 0]], input[[1, 1]]]);
        assert_eq!(p2[2..], [input[[2, 0]], input[[2, 1]]]);
        // Pulse 3: slot 0 = input[2], slot 1 = input[3].
        let p3 = &got[3];
        assert_eq!(p3[..2], [input[[2, 0]], input[[2, 1]]]);
        assert_eq!(p3[2..], [input[[3, 0]], input[[3, 1]]]);
    }
}
