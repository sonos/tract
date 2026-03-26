use crate::internal::*;
use crate::model::NonPulsingWrappingOp;
use tract_core::ops::array::{Slice, TypedConcat};
use tract_pulse_opl::concat::PulsedSameAxisConcat;
use tract_pulse_opl::ops::Delay;
use tract_pulse_opl::tract_core::ops::array::MultiBroadcastTo;
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
    ensure!(
        source_facts.iter().filter(|fact| fact.shape[axis].symbols().contains(symbol)).count() == 1,
        "Concat over pulse axis (#{axis}, {symbol:?}) expcts one single streaming input. Got: {source_facts:?}"
    );
    let pulsed_inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
    let pulse_facts: TVec<PulsedFact> = pulsed_inputs
        .iter()
        .map(|i| target.outlet_fact(*i).cloned())
        .collect::<TractResult<_>>()?;

    // Use the source model to identify which input carries the streaming symbol,
    // rather than the first pulsed fact with stream info. Inputs derived from the
    // stream (e.g. a slice selecting a single element) may also have stream info in
    // the pulsed model but their source shape is concrete, so they are treated as
    // pre/post rather than the main streaming input.
    let stream_input_ix =
        source_facts.iter().position(|f| f.shape[axis].symbols().contains(symbol)).unwrap();
    let pulse_fact = &pulse_facts[stream_input_ix];
    ensure!(
        pulse_fact.stream.is_some(),
        "Expected pulsed fact at stream_input_ix={stream_input_ix} to be streaming: {pulse_fact:?}"
    );
    let stream = pulse_fact.stream.as_ref().unwrap();

    // before_len and after_len come from the source model where non-streaming
    // inputs have concrete shapes.
    let before_len: usize = source_facts[..stream_input_ix]
        .iter()
        .map(|f| f.shape[axis].to_usize())
        .collect::<TractResult<Vec<_>>>()?
        .into_iter()
        .sum();
    let after_len: usize = source_facts[stream_input_ix + 1..]
        .iter()
        .map(|f| f.shape[axis].to_usize())
        .collect::<TractResult<Vec<_>>>()?
        .into_iter()
        .sum();

    let zero = target
        .add_const(format!("{name}.zero"), Tensor::zero_scalar_dt(source_facts[0].datum_type)?)?;
    let mut shape = pulse_fact.shape.clone();
    shape.set(axis, 0.to_dim());
    let empty = target.wire_node(
        format!("{name}.pre"),
        NonPulsingWrappingOp(Box::new(MultiBroadcastTo { shape })),
        &[zero],
    )?[0];

    // Build pre node: concat of pulsed inputs before the main stream, then sliced
    // to before_len so the runtime tensor has the right size at axis regardless of
    // whether the inputs are themselves pulsed (pulse-sized) or constant.
    let pre = if stream_input_ix > 0 {
        let pre_concat = target.wire_node(
            format!("{name}.pre"),
            NonPulsingWrappingOp(Box::new(TypedConcat::new(axis))),
            &pulsed_inputs.iter().take(stream_input_ix).cloned().collect_vec(),
        )?[0];
        target.wire_node(
            format!("{name}.pre.slice"),
            NonPulsingWrappingOp(Box::new(Slice {
                axis,
                start: 0.to_dim(),
                end: before_len.to_dim(),
            })),
            &[pre_concat],
        )?[0]
    } else {
        empty
    };

    // Build post node: same pattern, sliced to after_len.
    let post = if stream_input_ix + 1 < pulsed_inputs.len() {
        let post_concat = target.wire_node(
            format!("{name}.post"),
            NonPulsingWrappingOp(Box::new(TypedConcat::new(axis))),
            &pulsed_inputs.iter().skip(stream_input_ix + 1).cloned().collect_vec(),
        )?[0];
        target.wire_node(
            format!("{name}.post.slice"),
            NonPulsingWrappingOp(Box::new(Slice {
                axis,
                start: 0.to_dim(),
                end: after_len.to_dim(),
            })),
            &[post_concat],
        )?[0]
    } else {
        empty
    };

    let mut input = pulsed_inputs[stream_input_ix];
    // The main stream must arrive late enough that the pre content fits before it.
    // effective_delay = max(stream.delay, before_len).
    let effective_delay = if stream.delay < before_len {
        input = target.wire_node(
            format!("{}.Delay", node.name),
            Delay::new_typed(
                source.outlet_fact(node.inputs[stream_input_ix])?,
                stream.axis,
                before_len - stream.delay,
                0,
            ),
            &[input],
        )?[0];
        before_len
    } else {
        stream.delay
    };

    let main_op = PulsedSameAxisConcat {
        axis: op.axis,
        before_len,
        after_len,
        input_delay: effective_delay,
        input_len: stream.dim.clone(),
    };
    Ok(Some(target.wire_node(&*node.name, main_op, &[pre, input, post])?))
}

impl PulsedOp for PulsedSameAxisConcat {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let &[_pre, fact, _post] = inputs else { bail!("Expect 3 inputs") };
        let mut fact: PulsedFact = fact.clone();
        let stream = fact.stream.as_mut().unwrap();
        stream.dim += (self.before_len + self.after_len).to_dim();
        stream.delay -= self.before_len;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
