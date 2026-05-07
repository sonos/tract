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

    // Non-streaming input: two sub-cases.
    //
    //  - Session-buffered input (rel-pos table sliced to the current
    //    frame count): the input fact's shape *itself* contains the
    //    streaming symbol, signalling that this tensor is bound once
    //    per session with concretely-resolved `S`.  Keep the Slice's
    //    `start`/`end` *symbolic* — they'll evaluate against
    //    `session.resolved_symbols` at runtime, picking the right
    //    window for the resolved S.  Substituting S→pulse here would
    //    pin the slice to the per-pulse chunk count (= 1), which is
    //    wrong.
    //
    //  - Genuinely-static slice (e.g. a precomputed PE table whose
    //    *value* depends on S only through start/end): same as before,
    //    substitute S→pulse to concretize.
    if fact.stream.is_none() {
        let session_buffered = fact.shape.iter().any(|d| d.symbols().contains(symbol));
        if session_buffered {
            use crate::model::NonPulsingWrappingOp;
            let symbolic_op = NonPulsingWrappingOp(Box::new(op.clone()));
            return target.wire_node(&*node.name, symbolic_op, &[input]).map(Some);
        }
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
