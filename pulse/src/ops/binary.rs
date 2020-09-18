use crate::internal::*;
use tract_pulse_opl::ops::Delay;
use tract_core::ops::binary::*;

submit_op_pulsifier!(TypedBinOp, pulsify_bin);
submit_op_pulsifier!(UnaryOp, pulsify_un);

fn pulsify_bin(
    op: &TypedBinOp,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let delay = (0..2)
        .map(|ix| target.outlet_fact(mapping[&node.inputs[ix]]).unwrap().delay)
        .max()
        .unwrap();
    let mut inputs = tvec!();
    for ix in 0..2 {
        let mut input = mapping[&node.inputs[ix]];
        let fact = target.outlet_fact(input)?.clone();
        if fact.delay < delay {
            let add_delay = delay - fact.delay;
            input = target.wire_node(
                format!("{}.Delay", &*node.name),
                Delay::new(fact.axis, &fact.into(), add_delay, 0),
                &[input],
            )?[0];
        }
        inputs.push(input);
    }
    target.wire_node(&*node.name, op.clone(), &*inputs)
}

impl PulsedOp for TypedBinOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        fact.shape = tract_core::broadcast::multi_broadcast(&[&inputs[0].shape, &inputs[1].shape])
            .ok_or_else(|| {
                format!("Can not broadcast: {:?} and {:?}", inputs[0].shape, inputs[1].shape)
            })?;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

fn pulsify_un(
    op: &UnaryOp,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let input = mapping[&node.inputs[0]];
    target.wire_node(&*node.name, op.clone(), &[input])
}

impl PulsedOp for UnaryOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type =
            self.mini_op.result_datum_type(inputs[0].datum_type, self.a.datum_type())?;
        fact.shape = tract_core::broadcast::multi_broadcast(&[
            &inputs[0].shape,
            &self.a.shape().iter().map(|d| d.into()).collect(),
        ])
        .unwrap();
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
