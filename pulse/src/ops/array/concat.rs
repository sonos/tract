use crate::internal::*;
use crate::model::NonPulsingWrappingOp;
use tract_core::ops::array::TypedConcat;
use tract_pulse_opl::concat::PulsedSameAxisConcat;
use tract_pulse_opl::ops::Delay;
use tract_pulse_opl::tract_core::tract_data::itertools::Itertools;

register_all!(TypedConcat: pulsify);

fn pulsify(
    op: &TypedConcat,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let pulse_facts: TVec<PulsedFact> =
        node.inputs.iter().map(|i| target.outlet_fact(mapping[i]).unwrap().clone()).collect();
    let (_stream_input_ix, pulse_fact) =
        pulse_facts.iter().enumerate().find(|(_ix, pf)| pf.stream.is_some()).unwrap();

    if pulse_fact.stream.as_ref().unwrap().axis == op.axis {
        pulsify_along_concat_axis(op, source, node, target, mapping, symbol)
    } else {
        Ok(None)
    }
}

fn pulsify_along_concat_axis(
    op: &TypedConcat,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
) -> TractResult<Option<TVec<OutletId>>> {
    let name = &node.name;
    let axis = op.axis;
    let source_facts: TVec<TypedFact> =
        node.inputs.iter().map(|i| source.outlet_fact(*i).unwrap().clone()).collect();
    ensure!(source_facts.iter().filter(|fact| fact.shape[axis].symbols().contains(symbol)).count() == 1,
        "Concat over pulse axis (#{axis}, {symbol:?}) expcts one single streaming input. Got: {source_facts:?}"
    );
    let pulsed_inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
    let pulse_facts: TVec<PulsedFact> = pulsed_inputs
        .iter()
        .map(|i| target.outlet_fact(*i).cloned())
        .collect::<TractResult<_>>()?;
    let (stream_input_ix, pulse_fact) =
        pulse_facts.iter().enumerate().find(|(_ix, pf)| pf.stream.is_some()).unwrap();
    let stream = pulse_fact.stream.as_ref().unwrap();

    let pre = target.wire_node(
        format!("{name}.pre"),
        NonPulsingWrappingOp(Box::new(TypedConcat::new(axis))),
        &*pulsed_inputs.iter().take(stream_input_ix).cloned().collect_vec(),
    )?[0];
    let post = target.wire_node(
        format!("{name}.post"),
        NonPulsingWrappingOp(Box::new(TypedConcat::new(axis))),
        &*pulsed_inputs.iter().skip(stream_input_ix + 1).cloned().collect_vec(),
    )?[0];

    let mut input = pulsed_inputs[stream_input_ix];
    let pre_fact = target.outlet_fact(pre)?;
    let before = pre_fact.shape[op.axis].to_usize()?;
    if stream.delay < before {
        input = target.wire_node(
            format!("{}.Delay", node.name),
            Delay::new_typed(
                source.outlet_fact(node.inputs[stream_input_ix])?,
                stream.axis,
                before - stream.delay,
                0,
            ),
            &[input],
        )?[0];
    }
    let main_op = PulsedSameAxisConcat {
        axis: op.axis,
        input_delay: stream.delay.saturating_sub(before),
        input_len: stream.dim.clone(),
    };
    Ok(Some(target.wire_node(&*node.name, main_op, &[pre, input, post])?))
}

impl PulsedOp for PulsedSameAxisConcat {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let &[pre, fact, post] = inputs else { bail!("Expect 3 inputs") };
        let mut fact: PulsedFact = fact.clone();
        let stream = fact.stream.as_mut().unwrap();
        let before = pre.shape[self.axis].to_usize()?;
        let after = post.shape[self.axis].to_usize()?;
        stream.dim += (before + after).to_dim();
        stream.delay -= before.to_usize()?;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
