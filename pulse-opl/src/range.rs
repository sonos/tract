//! `PulsedRange` — pulsified `tract_core::ops::array::Range`.
//!
//! A 0-input streaming generator: at pulse `n` it emits
//! `[start + step·(n·P), …, start + step·((n+1)·P − 1)]` of length `P`
//! (the pulse size on the streaming axis).  State is just `current_pos`,
//! mirroring `PulsePadOpState`.
//!
//! The op is registered as a `PulsedOp` only — there's no NNEF/typed-side
//! representation (it lives strictly between the pulsifier and runtime).

use tract_nnef::internal::*;
use tract_nnef::tract_core::trivial_op_state_freeze;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PulsedRange {
    /// Output dtype (matches `start`'s dtype on the typed side).
    pub datum_type: DatumType,
    /// 0-d const `start` value, of dtype `datum_type` (TDim or i64/…).
    pub start: Tensor,
    /// 0-d const `step` value, of dtype `datum_type`.
    pub step: Tensor,
    /// Symbolic total length of the range (= `(end - start)/step`).
    pub stream_dim: TDim,
    /// Pulse size on the (single) streaming axis.
    pub pulse: usize,
}

impl Op for PulsedRange {
    fn name(&self) -> StaticName {
        "PulsedRange".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "dtype: {:?} start: {:?} step: {:?} dim: {} pulse: {}",
            self.datum_type, self.start, self.step, self.stream_dim, self.pulse,
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for PulsedRange {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<PulsedRangeState>::default()))
    }
}

impl TypedOp for PulsedRange {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.datum_type.fact([self.pulse.to_dim()])))
    }

    as_op!();
}

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
struct PulsedRangeState {
    current_pos: usize,
}

impl OpState for PulsedRangeState {
    fn eval(
        &mut self,
        session: &mut TurnState,
        op: &dyn Op,
        _inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<PulsedRange>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let pulse = op.pulse;
        let base = self.current_pos;
        self.current_pos += pulse;

        let tensor = if op.start.datum_type() == TDim::datum_type() {
            // TDim-input form: `Range::make` materialises as i64 regardless
            // of `op.datum_type` (which is what `Range::output_facts` writes
            // for the TDim-input branch — see `core/src/ops/array/range.rs`).
            let start = op
                .start
                .try_as_plain()?
                .to_scalar::<TDim>()?
                .eval(&session.resolved_symbols)
                .to_i64()?;
            let step = op
                .step
                .try_as_plain()?
                .to_scalar::<TDim>()?
                .eval(&session.resolved_symbols)
                .to_i64()?;
            let data: Vec<i64> =
                (0..pulse).map(|i| start + step * (base as i64 + i as i64)).collect();
            tract_nnef::tract_core::ndarray::Array1::from_vec(data).into_dyn().into_tensor()
        } else {
            dispatch_numbers!(make_pulse(op.datum_type)(&op.start, &op.step, base, pulse))?
        };

        Ok(tvec!(tensor.into_tvalue()))
    }
}

fn make_pulse<T>(start: &Tensor, step: &Tensor, base: usize, pulse: usize) -> TractResult<Tensor>
where
    T: Datum
        + Copy
        + tract_num_traits::NumCast
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>,
{
    let start = *start.try_as_plain()?.to_scalar::<T>()?;
    let step = *step.try_as_plain()?.to_scalar::<T>()?;
    let base_t: T = tract_num_traits::cast(base as i64)
        .ok_or_else(|| format_err!("PulsedRange: base {base} doesn't fit in target dtype"))?;
    let mut data = Vec::with_capacity(pulse);
    let mut v = start + step * base_t;
    for _ in 0..pulse {
        data.push(v);
        v = v + step;
    }
    Ok(tract_nnef::tract_core::ndarray::Array1::from_vec(data).into_dyn().into_tensor())
}

trivial_op_state_freeze!(PulsedRangeState);
