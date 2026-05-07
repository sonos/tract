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

    // Non-streaming input: defer the slice to session start, keeping
    // `start`/`end` symbolic whenever they reference the streaming
    // symbol so they evaluate against `session.resolved_symbols` rather
    // than getting pinned to `S=pulse` (pulse=1 in chunked frames).
    //
    //  - Input shape itself contains the streaming symbol
    //    (session-buffered Source, e.g. ex12's pos external).
    //  - Or the bounds reference the streaming symbol while the input
    //    is concretely shaped (encoder.p1's `posEnc_posEmb_dyn_slice`
    //    of a 5000-row constant: input is `[5000, 1024]` constant but
    //    `start = 4999 - (T+6)/8`, `end = 5000 + (T+6)/8` are
    //    streaming-symbol-dependent — output should be
    //    `[1+2·(T+6)/8, 1024]` resolved at session start).
    //
    // Both routes wire the original `Slice` symbolically through
    // `NonPulsingWrappingOp`; the runtime evaluates `start`/`end` once
    // per session.  Substituting `S → pulse` here would lock the
    // window to `S=1`, breaking session-buffered semantics for any
    // downstream that needs the full resolved-S window (e.g.
    // blockify's rel-pos `pos_slice` reading
    // `[k·S − (L+1)·k, k·S + k − 1]`).
    if fact.stream.is_none() {
        let in_shape_session = fact.shape.iter().any(|d| d.symbols().contains(symbol));
        let bounds_session =
            op.start.symbols().contains(symbol) || op.end.symbols().contains(symbol);
        if in_shape_session || bounds_session {
            use crate::model::NonPulsingWrappingOp;
            let symbolic_op = NonPulsingWrappingOp(Box::new(op.clone()));
            return target.wire_node(&*node.name, symbolic_op, &[input]).map(Some);
        }
        // Both shape and bounds are concrete in the streaming symbol —
        // the slice can be wired as-is (still through
        // `NonPulsingWrappingOp` since the input doesn't stream).
        use crate::model::NonPulsingWrappingOp;
        let concrete_op = NonPulsingWrappingOp(Box::new(op.clone()));
        return target.wire_node(&*node.name, concrete_op, &[input]).map(Some);
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
