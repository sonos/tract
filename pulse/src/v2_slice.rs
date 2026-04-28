/// PulseV2 handling for Slice on the streaming axis.
///
/// Under fixed-pulse semantics, Slice on the streaming axis is purely
/// bookkeeping: per-pulse output shape equals input shape (constant `P`),
/// the data passes through unchanged. The slice's `start` is the v2-flavoured
/// equivalent of v1's `delay` — it shifts the output stream's effective
/// position relative to the input stream — and it's tracked at sink/merge
/// time, not in per-pulse tensor sizes. The op exists at all so consumers
/// (e.g. test harness, downstream merges) can read the `start`/`end`
/// metadata off the graph; it shunts itself out at declutter time.
///
/// Slice on a non-streaming axis is left alone — the original `Slice` op
/// runs normally there.
use crate::internal::*;
use crate::v2::{AxisRegion, PulseV2Action, PulseV2Region, PulseV2Symbols, RegionTransform};
use tract_pulse_opl::tract_core::ops::array::Slice;

fn slice_transform(
    op: &dyn TypedOp,
    source_region: &PulseV2Region,
    _symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let slice = op.downcast_ref::<Slice>().unwrap();
    let on_streaming_axis = source_region
        .axes
        .get(slice.axis)
        .is_some_and(|a| matches!(a, AxisRegion::Streaming { .. }));
    if !on_streaming_axis {
        // Non-streaming axis: original Slice op behaves correctly under
        // fixed-pulse, no replacement needed.
        return Ok(None);
    }
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

    /// At declutter time the metadata role is over (any consumer that needed
    /// to read `start`/`end` already had its chance during pulsification);
    /// shunt the op so the resulting graph matches v1's topology — Buffer/
    /// Delay only, no streaming-axis Slice nodes.
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        TypedModelPatch::shunt_one_op(model, node)
    }
}
