use crate::internal::*;
use tract_core::ops::array::Slice;

register_all!(Slice: pulsify);

fn pulsify(
    op: &Slice,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    let stream = fact.stream.with_context(|| {
        format!(
            "Unexpected streamless fact in pulsify {node}\ninput:{:?}",
            target.outlet_fact(input).unwrap()
        )
    })?;
    if op.axis == stream.axis {
        let skip = op.start.to_usize()?;
        let take = (op.end.clone() - &op.start).to_dim();
        let op = PulsedAxisSlice { axis: op.axis, skip, take };
        Ok(Some(target.wire_node(&*node.name, op, &[input])?))
    } else {
        Ok(None)
    }
}

#[derive(Debug, Clone, Default, Hash)]
pub struct PulsedAxisSlice {
    pub axis: usize,
    pub skip: usize,
    pub take: TDim,
}

impl Op for PulsedAxisSlice {
    fn name(&self) -> Cow<str> {
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
