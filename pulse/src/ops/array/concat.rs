use crate::internal::*;
use tract_core::ops::array::TypedConcat;
use tract_pulse_opl::concat::{overwrite_part_of_pulse, PulsedSameAxisConcat};
use tract_pulse_opl::ops::Delay;
use tract_pulse_opl::tract_core::trivial_op_state_freeeze;

register_all!(TypedConcat: pulsify);

fn pulsify(
    op: &TypedConcat,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let pulse_facts: TVec<PulsedFact> =
        node.inputs.iter().map(|i| target.outlet_fact(mapping[i]).unwrap().clone()).collect();
    let (_stream_input_ix, pulse_fact) =
        pulse_facts.iter().enumerate().find(|(_ix, pf)| pf.stream.is_some()).unwrap();

    if pulse_fact.stream.as_ref().unwrap().axis == op.axis {
        pulsify_along_concat_axis(op, source, node, target, mapping)
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
) -> TractResult<Option<TVec<OutletId>>> {
    let axis = op.axis;
    let source_facts: TVec<TypedFact> =
        node.inputs.iter().map(|i| source.outlet_fact(*i).unwrap().clone()).collect();
    if !source_facts.iter().all(|f| f.konst.is_some() || f.shape[axis].to_usize().is_err()) {
        bail!("Concat over pulse axis (#{axis}) expect constant input and one stream input. Got: {source_facts:?}")
    }
    let pulse_facts: TVec<PulsedFact> =
        node.inputs.iter().map(|i| target.outlet_fact(mapping[i]).unwrap().clone()).collect();
    let (stream_input_ix, pulse_fact) =
        pulse_facts.iter().enumerate().find(|(_ix, pf)| pf.stream.is_some()).unwrap();
    let stream = pulse_fact.stream.as_ref().unwrap();

    let pre_owned = source_facts
        .iter()
        .take(stream_input_ix)
        .map(|s| s.konst.clone().unwrap())
        .collect::<TVec<_>>();
    let pre = Tensor::stack_tensors(op.axis, &pre_owned)?;
    let post_owned = source_facts
        .iter()
        .skip(stream_input_ix + 1)
        .map(|s| s.konst.clone().unwrap())
        .collect::<TVec<_>>();
    let post = Tensor::stack_tensors(op.axis, &post_owned)?;

    let mut input = mapping[&node.inputs[stream_input_ix]];
    let before = pre.shape()[op.axis];
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
        pre_slice: pre,
        post_slice: post,
        input_delay: stream.delay.saturating_sub(before),
        input_len: stream.dim.clone(),
    };
    Ok(Some(target.wire_node(&*node.name, main_op, &[input])?))
}

impl PulsedOp for PulsedSameAxisConcat {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let stream = fact.stream.as_mut().unwrap();
        let before = self.pre_slice.shape()[self.axis];
        let after = self.post_slice.shape()[self.axis];
        stream.dim += (before + after).to_dim();
        stream.delay -= before;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
