use crate::internal::*;
use tract_core::ops::array::Slice;

register_all!(Slice: pulsify);

fn pulsify(
    op: &Slice,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();

    // Non-streaming input: the streaming symbol appears only in start/end
    // (e.g. a static PE table sliced to the current frame count, or to a
    // symmetric RPE window centered at MAX with start=MAX-T', end=MAX+T'+1).
    // Substitute S→pulse directly to get the concrete per-pulse bounds.
    if fact.stream.is_none() {
        let start = op.start.substitute(symbol, pulse)?;
        let end = op.end.substitute(symbol, pulse)?;
        if start.symbols().is_empty() && end.symbols().is_empty() {
            use crate::model::NonPulsingWrappingOp;
            let concrete_op = NonPulsingWrappingOp(Box::new(Slice { axis: op.axis, start, end }));
            return target.wire_node(&*node.name, concrete_op, &[input]).map(Some);
        }
        return Ok(None);
    }

    let stream = fact.stream.as_ref().unwrap();
    if op.axis == stream.axis {
        let start = op.start.substitute(symbol, pulse)?;
        let skip = start.to_usize()?;
        let take = node.outputs[0].fact.shape[op.axis].clone();
        let op = PulsedAxisSlice { axis: op.axis, skip, take };
        Ok(Some(target.wire_node(&*node.name, op, &[input])?))
    } else {
        // Slice on a non-streaming axis whose bounds may contain the streaming
        // symbol (e.g. axis 2 sliced to 2*T'-1 after the skew reshape).
        //
        // Try full substitution first (S→pulse): correct for bounds like
        // MAX-T' / MAX+T'+1 (RPE symmetric window) where the constant base
        // term must be preserved.
        //
        // If full substitution leaves symbols (e.g. TDim::Broadcast artifacts
        // from shape_of chains), fall back to the boundary-correction delta
        // formula (sub(S,pulse) - sub(S,0)) which cancels those artifacts.
        let start_full = op.start.substitute(symbol, pulse)?;
        let end_full = op.end.substitute(symbol, pulse)?;
        if start_full.symbols().is_empty() && end_full.symbols().is_empty() {
            use crate::model::PulseWrappingOp;
            return Ok(Some(target.wire_node(
                &*node.name,
                PulseWrappingOp(Box::new(Slice {
                    axis: op.axis,
                    start: start_full,
                    end: end_full,
                })),
                &[input],
            )?));
        }
        // Full substitution left symbols; try delta formula to cancel artifacts.
        let start = start_full - op.start.substitute(symbol, &TDim::Val(0))?;
        let end = end_full - op.end.substitute(symbol, &TDim::Val(0))?;
        if start.symbols().is_empty() && end.symbols().is_empty() {
            use crate::model::PulseWrappingOp;
            return Ok(Some(target.wire_node(
                &*node.name,
                PulseWrappingOp(Box::new(Slice { axis: op.axis, start, end })),
                &[input],
            )?));
        }
        Ok(None)
    }
}

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct PulsedAxisSlice {
    pub axis: usize,
    pub skip: usize,
    pub take: TDim,
}

impl Op for PulsedAxisSlice {
    fn name(&self) -> StaticName {
        "PulsedAxisSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis:{}, skip:{} take:{}", self.axis, self.skip, self.take)])
    }

    not_a_typed_op!();
}

impl EvalOp for PulsedAxisSlice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }
}

impl PulsedOp for PulsedAxisSlice {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let stream = fact.stream.as_mut().unwrap();
        stream.delay += self.skip;
        stream.dim = self.take.clone();
        Ok(tvec!(fact))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::<tract_pulse_opl::tract_core::ops::identity::Identity>::default()
    }

    as_op!();
}
