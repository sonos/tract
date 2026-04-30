//! `WindowOnAxis` — typed op produced by Blockify for banded-mask sections.
//!
//! Inserts a static `window` axis after the streaming `axis`, where each
//! window slot `w ∈ [0, W)` carries a `start`-shifted view of the input:
//!
//! ```text
//! input  shape: [..., S, ...]                       (S at `axis`)
//! output shape: [..., S, W, ...]                    (W inserted after `axis`)
//! output[..., s, w, ...] = input[..., s + start + w, ...]
//!                          if 0 ≤ s + start + w < S
//!                          0 otherwise
//! ```
//!
//! `start = 0` is the future window (slot 0 = current, ex03).  `start = -(W-1)`
//! is the past window (slot W-1 = current, ex04).  Mixed values are allowed
//! provided `start ≤ 0 ≤ start + W - 1` (the window straddles current).
//!
//! Pulsification (in `tract_pulse::ops::window`) emits `Delay(0, W-1)` plus a
//! per-pulse reshape that splits the streaming-axis pulse into `[1, W]` and
//! retimes `stream.delay` by `-start` so the consumer's logical time anchors
//! to the *latest* chunk in the buffer when `start < 0` (vs the oldest when
//! `start = 0`).  The op only ever lives between the Blockify rewrite pass
//! and pulsification.

use tract_nnef::internal::*;
use tract_nnef::tract_core::ndarray::{ArrayD, Axis, Slice};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowOnAxis {
    pub axis: usize,
    pub window: usize,
    /// Offset of slot 0 relative to the streaming index `s`.  Default 0.
    /// Negative values shift the window into the past (slot 0 looks
    /// backward); positive values shift into the future (skipping current).
    pub start: i64,
}

impl WindowOnAxis {
    /// Convenience constructor for the future-window form (slot 0 = current).
    pub fn future(axis: usize, window: usize) -> Self {
        Self { axis, window, start: 0 }
    }

    fn eval_t<T>(&self, input: TValue) -> TractResult<TValue>
    where
        T: Datum + Copy + Default,
    {
        let input = input.to_plain_array_view::<T>()?;
        let s = input.shape()[self.axis] as i64;
        let mut out_shape: Vec<usize> = input.shape().to_vec();
        out_shape.insert(self.axis + 1, self.window);
        let mut output: ArrayD<T> = ArrayD::from_elem(out_shape, T::default());
        for w in 0..self.window {
            let shift = self.start + w as i64;
            // For each w: output[s, w, …] = input[s + shift, …] when in
            // bounds.  Valid `s` range and corresponding source indices:
            //   shift ≥ 0: src [shift, S),    dst [0, S-shift)
            //   shift < 0: src [0, S+shift),  dst [-shift, S)
            let src_start = shift.max(0).min(s) as usize;
            let src_end = (s + shift.min(0)).max(0).min(s) as usize;
            if src_start >= src_end {
                continue;
            }
            let dst_start = (-shift).max(0).min(s) as usize;
            let dst_end = dst_start + (src_end - src_start);
            let src = input.slice_axis(Axis(self.axis), Slice::from(src_start..src_end));
            let mut dst_w = output.index_axis_mut(Axis(self.axis + 1), w);
            dst_w.slice_axis_mut(Axis(self.axis), Slice::from(dst_start..dst_end)).assign(&src);
        }
        Ok(output.into_tensor().into_tvalue())
    }
}

impl Op for WindowOnAxis {
    fn name(&self) -> StaticName {
        "WindowOnAxis".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {} window: {} start: {}", self.axis, self.window, self.start)])
    }

    op_as_typed_op!();
}

impl EvalOp for WindowOnAxis {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_numbers!(Self::eval_t(input.datum_type())(self, input))?))
    }
}

impl TypedOp for WindowOnAxis {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.axis < inputs[0].rank(), "WindowOnAxis: axis {} out of range", self.axis);
        ensure!(self.window > 0, "WindowOnAxis: window must be > 0");
        let mut shape: TVec<TDim> = inputs[0].shape.iter().cloned().collect();
        shape.insert(self.axis + 1, self.window.to_dim());
        let mut fact = inputs[0].without_value();
        fact.shape = shape.into();
        Ok(tvec!(fact))
    }
}
