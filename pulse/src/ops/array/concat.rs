use crate::internal::*;
use tract_core::ops::array::TypedConcat;
use tract_pulse_opl::concat::overwrite_part_of_pulse;
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
        bail!("Concat over pulse axis expect constant input and one stream input")
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

/// Concat with pulse along concat axis
#[derive(Debug, Clone, Hash)]
pub struct PulsedSameAxisConcat {
    axis: usize,
    pre_slice: Tensor,
    post_slice: Tensor,
    input_delay: usize,
    input_len: TDim,
}

impl Op for PulsedSameAxisConcat {
    fn name(&self) -> Cow<str> {
        "PulsedSameAxisConcat".into()
    }

    op_as_typed_op!();
}

impl EvalOp for PulsedSameAxisConcat {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        let symbols_in_dim = self.input_len.symbols().into_iter().collect();
        Ok(Some(Box::new(PulsedSameAxisConcatState { current_pos: 0, symbols_in_dim })))
    }
}

impl TypedOp for PulsedSameAxisConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }
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

#[derive(Clone, Debug)]
pub struct PulsedSameAxisConcatState {
    current_pos: usize,
    symbols_in_dim: Vec<Symbol>,
}
trivial_op_state_freeeze!(PulsedSameAxisConcatState);

impl OpState for PulsedSameAxisConcatState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op
            .downcast_ref::<PulsedSameAxisConcat>()
            .ok_or_else(|| format_err!("Wrong Op type"))?;
        let input = args_1!(inputs);
        let mut data = input.into_tensor();
        let pulse = data.shape()[op.axis];
        let current_pos = self.current_pos;
        self.current_pos += pulse;

        let pre_length = op.pre_slice.shape()[op.axis];
        let pre_offset = op.input_delay - pre_length;
        overwrite_part_of_pulse(op.axis, &mut data, current_pos, &op.pre_slice, pre_offset)?;
        if self.symbols_in_dim.iter().all(|s| session.resolved_symbols.get(s).is_some()) {
            let l = op.input_len.eval(&session.resolved_symbols).to_usize().unwrap();
            let post_offset = op.input_delay + l;
            overwrite_part_of_pulse(op.axis, &mut data, current_pos, &op.post_slice, post_offset)?;
        }

        Ok(tvec!(data.into_tvalue()))
    }
}
