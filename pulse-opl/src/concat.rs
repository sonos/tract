use std::ops::Range;
use tract_nnef::internal::*;
use tract_nnef::tract_core::trivial_op_state_freeeze;

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
        true
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<PulsedSameAxisConcatState>::default()))
    }
}

impl TypedOp for PulsedSameAxisConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct PulsedSameAxisConcatState {
    current_pos: usize,
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
        if let Ok(l) = op.input_len.eval(&session.resolved_symbols).to_usize() {
            let post_offset = op.input_delay + l;
            overwrite_part_of_pulse(op.axis, &mut data, current_pos, &op.post_slice, post_offset)?;
        }

        Ok(tvec!(data.into_tvalue()))
    }
}

pub fn overwrite_part_of_pulse(
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

    match range_in_range(&pulse_range, &const_range) {
        RangeInRange::Before(_) | RangeInRange::After(_) => (),
        RangeInRange::Begin(offset) => {
            // ----[<----->HHH]HH----
            pulse_data.assign_slice(offset..pulse, const_data, 0..pulse - offset, axis)?;
        }
        RangeInRange::Contain(offset) => {
            // ----[<----->HHHHHHH-]---
            pulse_data.assign_slice(
                offset..offset + const_length,
                const_data,
                0..const_length,
                axis,
            )?;
        }
        RangeInRange::Inside(offset) => {
            // ----------<H>[HH]HH----
            pulse_data.assign_slice(0..pulse, const_data, offset..offset + pulse, axis)?;
        }
        RangeInRange::End(offset) => {
            // --------<HHH>[HHHH-]---
            pulse_data.assign_slice(
                0..const_length - offset,
                const_data,
                offset..const_length,
                axis,
            )?;
        }
    }
    Ok(())
}

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub enum RangeInRange {
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

pub fn range_in_range(needle: &Range<usize>, haystack: &Range<usize>) -> RangeInRange {
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
