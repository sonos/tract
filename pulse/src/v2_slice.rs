/// PulseV2 handling for Slice on the streaming axis.
///
/// Under fixed-pulse semantics, Slice on the streaming axis is purely
/// bookkeeping: per-pulse output shape equals input shape (constant `P`),
/// the data passes through unchanged. The slice's `start` is the v2-flavoured
/// equivalent of v1's `delay` — it shifts the output stream's effective
/// position relative to the input stream — and it's tracked at sink/merge
/// time, not in per-pulse tensor sizes.
///
/// The slice's `end` bounds the total output stream length, which the
/// fixed-pulse runtime ignores per pulse (we run with garbage past end and
/// truncate at the sink against the symbol-resolved stream length).
use crate::internal::*;
use crate::v2::{PulseV2Action, PulseV2Region, PulseV2Symbols, RegionTransform};
use tract_pulse_opl::tract_core::ops::array::Slice;

fn slice_transform(
    op: &dyn TypedOp,
    _source_region: &PulseV2Region,
    _symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let slice = op.downcast_ref::<Slice>().unwrap();
    Ok(Some(PulseV2Action::ReplaceOp(Box::new(PulseV2Slice {
        axis: slice.axis,
        start: slice.start.clone(),
        end: slice.end.clone(),
    }))))
}

inventory::submit! {
    RegionTransform { type_id: std::any::TypeId::of::<Slice>(), func: slice_transform }
}

/// Pass-through op on the streaming axis. `start` and `end` are kept as
/// metadata so the harness can account for the delay shift.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PulseV2Slice {
    pub axis: usize,
    pub start: TDim,
    pub end: TDim,
}

impl Op for PulseV2Slice {
    fn name(&self) -> StaticName {
        "PulseV2Slice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis={} [{}, {})", self.axis, self.start, self.end)])
    }

    op_as_typed_op!();
}

impl EvalOp for PulseV2Slice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        _session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }
}

impl TypedOp for PulseV2Slice {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }
}
