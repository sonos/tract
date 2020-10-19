use crate::internal::*;
use tract_core::ops::array::Slice;

submit_op_pulsifier!(Slice<TDim>, pulsify::<TDim>);
submit_op_pulsifier!(Slice<usize>, pulsify::<usize>);

fn pulsify<D: DimLike + ToDim + Hash>(
    op: &Slice<D>,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    let op: Box<dyn PulsedOp> = if op.axis == fact.axis {
        let skip = op.start.to_usize()?;
        let take = (op.end.clone() - &op.start).to_dim();
        PulsedAxisSlice { axis: op.axis, skip, take }.into()
    } else {
        tract_core::dyn_clone::clone_box(op)
    };
    target.wire_node(&*node.name, op, &[input])
}

impl<D: DimLike + ToDim + Hash> PulsedOp for Slice<D> {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let len = (self.end.clone() - &self.start).to_dim();
        if self.axis == fact.axis {
            fact.delay += self.start.to_usize()?;
            fact.dim = len
        } else {
            fact.shape[self.axis] = len;
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

#[derive(Debug, Clone, Default, Hash)]
pub struct PulsedAxisSlice {
    pub axis: usize,
    pub skip: usize,
    pub take: TDim,
}

tract_data::impl_dyn_hash!(PulsedAxisSlice);

impl Op for PulsedAxisSlice {
    fn name(&self) -> Cow<str> {
        "PulsedAxisSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis:{}, skip:{} take:{}", self.axis, self.skip, self.take)])
    }

    op_pulse!();
    not_a_typed_op!();
}

impl EvalOp for PulsedAxisSlice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(inputs)
    }
}

impl PulsedOp for PulsedAxisSlice {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.delay += self.skip;
        fact.dim = self.take.clone();
        Ok(tvec!(fact))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(tract_core::ops::identity::Identity::default())
    }

    as_op!();
}
