use crate::internal::*;
use tract_core::ops::binary::*;
use tract_core::ops::logic::Iff;
use tract_pulse_opl::ops::Delay;

submit_op_pulsifier!(UnaryOp, pulsify_un);
submit_op_pulsifier!(TypedBinOp, pulsify_bin);
submit_op_pulsifier!(Iff, pulsify_iff);

fn sync_inputs(
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<TVec<OutletId>> {
    let delay = node
        .inputs
        .iter()
        .map(|input| target.outlet_fact(mapping[input]).unwrap().delay)
        .max()
        .unwrap();
    let mut inputs = tvec!();
    for input in &node.inputs {
        let mut input = mapping[input];
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
    Ok(inputs)
}

fn pulsify_bin(
    op: &TypedBinOp,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let inputs = &*sync_inputs(node, target, mapping)?;
    target.wire_node(&*node.name, op.clone(), &inputs)
}

impl PulsedOp for TypedBinOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        fact.shape = tract_core::broadcast::multi_broadcast(&[&inputs[0].shape, &inputs[1].shape])
            .ok_or_else(|| {
                format_err!("Can not broadcast: {:?} and {:?}", inputs[0].shape, inputs[1].shape)
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

fn pulsify_iff(
    op: &Iff,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let inputs = &*sync_inputs(node, target, mapping)?;
    target.wire_node(&*node.name, op.clone(), &inputs)
}

impl PulsedOp for Iff {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = inputs[1].datum_type;
        fact.shape = tract_core::broadcast::multi_broadcast(&[
            &inputs[0].shape,
            &inputs[1].shape,
            &inputs[2].shape,
        ])
        .ok_or_else(|| {
            format_err!(
                "Can not broadcast: {:?}, {:?}, {:?}",
                inputs[0].shape,
                inputs[1].shape,
                inputs[2].shape
            )
        })?;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
