/// PulseV2 handling for Slice on the streaming axis.
///
/// In the pulsed model, Slice(start, end) becomes PulseV2Slice which,
/// at each pulse T, outputs the intersection of the pulse [T*P, (T+1)*P)
/// with the batch slice [start, end). The output size varies per pulse.
use crate::internal::*;
use crate::v2::{PulseV2Action, PulseV2Region, PulseV2Symbols, RegionTransform};
use tract_pulse_opl::tract_core::ops::array::Slice;

fn slice_transform(
    op: &dyn TypedOp,
    _source_region: &PulseV2Region,
    symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let slice = op.downcast_ref::<Slice>().unwrap();
    Ok(Some(PulseV2Action::ReplaceOp(Box::new(PulseV2Slice {
        axis: slice.axis,
        start: slice.start.clone(),
        end: slice.end.clone(),
        pulse_id: symbols.pulse_id.clone(),
    }))))
}

inventory::submit! {
    RegionTransform {
        type_id: std::any::TypeId::of::<Slice>(),
        func: slice_transform,
    }
}

/// Runtime op that trims each pulse to the intersection with [start, end).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PulseV2Slice {
    pub axis: usize,
    pub start: TDim,
    pub end: TDim,
    pub pulse_id: Symbol,
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
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let pulse_size = input.shape()[self.axis];

        let t = session.resolved_symbols.get(&self.pulse_id).unwrap_or(0) as usize;
        let start = self.start.eval(&session.resolved_symbols).to_i64()? as usize;
        let end = self.end.eval(&session.resolved_symbols).to_i64()? as usize;

        let pulse_offset = t * pulse_size;
        let inter_start = start.max(pulse_offset);
        let inter_end = end.min(pulse_offset + pulse_size);

        if inter_start >= inter_end {
            let mut shape = input.shape().to_vec();
            shape[self.axis] = 0;
            let empty = Tensor::zero_dt(input.datum_type(), &shape)?;
            return Ok(tvec!(empty.into_tvalue()));
        }

        let local_start = inter_start - pulse_offset;
        let local_end = inter_end - pulse_offset;
        let sliced = input.slice(self.axis, local_start, local_end)?;
        Ok(tvec!(sliced.into_tvalue()))
    }
}

impl TypedOp for PulseV2Slice {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        // Output size = max(0, min((T+1)*P, end) - max(T*P, start))
        // where T*P is the pulse start and (T+1)*P is the pulse end.
        let input_dim = inputs[0].shape[self.axis].clone();
        let t_sym = self.pulse_id.clone();
        let t = TDim::Sym(t_sym);
        let pulse_start = t.clone() * input_dim.clone();
        let pulse_end = (t + 1) * input_dim;
        let inter_start = TDim::Max(vec![pulse_start, self.start.clone()]);
        let inter_end = TDim::Min(vec![pulse_end, self.end.clone()]);
        let out_dim = TDim::Max(vec![TDim::Val(0), inter_end - inter_start]);
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, out_dim);
        Ok(tvec!(fact))
    }
}
