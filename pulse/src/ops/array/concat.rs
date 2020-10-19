use crate::internal::*;
use std::ops::Range;
use tract_core::ndarray::*;
use tract_core::ops::array::TypedConcat;
use tract_pulse_opl::ops::Delay;

submit_op_pulsifier!(TypedConcat, pulsify);

fn pulsify(
    op: &TypedConcat,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    if node.inputs.len() > 1 {
        bail!("Pulsification not implemented for more than one input to Concat")
    }

    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?;

    if fact.axis == op.axis {
        pulsify_along_concat_axis(op, source, node, target, mapping)
    } else {
        bail!("Pulsify for Concat on a separate axis is not implemented (but possible)");
    }
}

fn pulsify_along_concat_axis(
    op: &TypedConcat,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<TVec<OutletId>> {
    if node.inputs.len() > 1 {
        bail!("Concat can not pulse more than on input on concat axis")
    }
    let mut input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    assert_eq!(fact.axis, op.axis);
    let var_index = op.slices.iter().position(|s| s.is_var()).unwrap();
    let pre_owned = op.slices[0..var_index]
        .iter()
        .map(|s| s.as_const().unwrap().cast_to_dt(fact.datum_type))
        .collect::<TractResult<TVec<_>>>()?;
    let pre = Tensor::stack_tensors(op.axis, &*pre_owned)?;
    let post_owned = op.slices[var_index + 1..]
        .iter()
        .map(|s| s.as_const().unwrap().cast_to_dt(fact.datum_type))
        .collect::<TractResult<TVec<_>>>()?;
    let post = Tensor::stack_tensors(op.axis, &*post_owned)?;

    let before = pre.shape()[op.axis];
    if fact.delay < before {
        input = target.wire_node(
            format!("{}.Delay", node.name),
            Delay::new(fact.axis, &(&fact).into(), before - fact.delay, 0),
            &[input],
        )?[0];
    }
    let main_op = PulsedSameAxisConcat {
        axis: op.axis,
        pre_slice: pre,
        post_slice: post,
        input_delay: fact.delay.saturating_sub(before),
        input_len: fact.dim.clone(),
    };
    target.wire_node(&*node.name, main_op, &[input])
}

impl PulsedOp for PulsedSameAxisConcat {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let before = self.pre_slice.shape()[self.axis];
        let after = self.post_slice.shape()[self.axis];
        fact.dim += (before + after).to_dim();
        fact.delay -= before;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
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
tract_data::impl_dyn_hash!(PulsedSameAxisConcat);

impl Op for PulsedSameAxisConcat {
    fn name(&self) -> Cow<str> {
        "PulsedSameAxisConcat".into()
    }

    op_pulse!();
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
        return Ok(Some(Box::new(PulsedSameAxisConcatState { current_pos: 0, symbols_in_dim })));
    }
}

impl TypedOp for PulsedSameAxisConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

#[derive(Clone, Debug)]
pub struct PulsedSameAxisConcatState {
    current_pos: usize,
    symbols_in_dim: Vec<Symbol>,
}

impl OpState for PulsedSameAxisConcatState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op = op.downcast_ref::<PulsedSameAxisConcat>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let input = args_1!(inputs);
        let mut data = input.into_tensor();
        let pulse = data.shape()[op.axis];
        let current_pos = self.current_pos;
        self.current_pos += pulse;

        let pre_length = op.pre_slice.shape()[op.axis];
        let pre_offset = op.input_delay - pre_length;
        dispatch_datum!(overwrite_part_of_pulse(data.datum_type())(
            op.axis,
            &mut data,
            current_pos,
            &op.pre_slice,
            pre_offset
        ))?;
        if self.symbols_in_dim.iter().all(|s| session.resolved_symbols[*s].is_some()) {
            let l = op.input_len.eval(&session.resolved_symbols).to_usize().unwrap();
            let post_offset = op.input_delay + l as usize;
            dispatch_datum!(overwrite_part_of_pulse(data.datum_type())(
                op.axis,
                &mut data,
                current_pos,
                &op.post_slice,
                post_offset
            ))?;
        }

        return Ok(tvec!(data.into_arc_tensor()));
    }
}

pub fn overwrite_part_of_pulse<T: Datum>(
    axis: usize,
    pulse_data: &mut Tensor,
    current_pos: usize,
    const_data: &Tensor,
    const_offset: usize,
) -> TractResult<()> {
    let pulse = pulse_data.shape()[axis];
    let const_length = const_data.shape()[axis];
    let const_range = const_offset..const_offset + const_length;
    let pulse_range = current_pos..current_pos + pulse;
    let axis = Axis(axis);
    let mut pulse_data = pulse_data.to_array_view_mut::<T>()?;
    let const_data = const_data.to_array_view::<T>()?;

    match range_in_range(&pulse_range, &const_range) {
        RangeInRange::Before(_) | RangeInRange::After(_) => (),
        RangeInRange::Begin(offset) => {
            // ----[<----->HHH]HH----
            pulse_data
                .slice_axis_mut(axis, (offset..pulse).into())
                .assign(&const_data.slice_axis(axis, (0..pulse - offset).into()));
        }
        RangeInRange::Contain(offset) => {
            // ----[<----->HHHHHHH-]---
            pulse_data
                .slice_axis_mut(axis, (offset..offset + const_length).into())
                .assign(&const_data);
        }
        RangeInRange::Inside(offset) => {
            // ----------<H>[HH]HH----
            pulse_data.assign(&const_data.slice_axis(axis, (offset..offset + pulse).into()));
        }
        RangeInRange::End(offset) => {
            // --------<HHH>[HHHH-]---
            pulse_data
                .slice_axis_mut(axis, (0..const_length - offset).into())
                .assign(&const_data.slice_axis(axis, (offset..const_length).into()));
        }
    }
    Ok(())
}

#[derive(Copy, Clone, Debug)]
enum RangeInRange {
    /// ----[--]<-->HHHH----
    Before(usize),
    /// ----[<----->HHH]HH----
    Begin(usize),
    /// ----[<----->HHHHHHH-]---
    Contain(usize),
    /// ----------<H>[HH]HH----
    Inside(usize),
    /// --------<HHH>[HHHH-]---
    End(usize),
    /// --------HHHHHHH<->[--]---
    After(usize),
}

fn range_in_range(needle: &Range<usize>, haystack: &Range<usize>) -> RangeInRange {
    if needle.end <= haystack.start {
        RangeInRange::Before(haystack.start - needle.end)
    } else if needle.start < haystack.start {
        if needle.end < haystack.end {
            RangeInRange::Begin(haystack.start - needle.start)
        } else {
            RangeInRange::Contain(haystack.start - needle.start)
        }
    } else if needle.start >= haystack.end {
        RangeInRange::After(needle.start - haystack.end)
    } else if needle.end > haystack.end {
        RangeInRange::End(needle.start - haystack.start)
    } else {
        RangeInRange::Inside(needle.start - haystack.start)
    }
}
