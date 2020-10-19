use std::ops::Range;
use tract_core::ndarray::*;
use tract_nnef::internal::*;

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
        true
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        return Ok(Some(Box::new(PulsedSameAxisConcatState::default())));
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

        unsafe {
            let pre_length = op.pre_slice.shape()[op.axis];
            let pre_offset = op.input_delay - pre_length;
            dispatch_datum_by_size!(overwrite_part_of_pulse(data.datum_type())(
                op.axis,
                &mut data,
                current_pos,
                &op.pre_slice,
                pre_offset
            ));
            if let Ok(l) = op.input_len.eval(&session.resolved_symbols).to_usize() {
                let post_offset = op.input_delay + l as usize;
                dispatch_datum_by_size!(overwrite_part_of_pulse(data.datum_type())(
                    op.axis,
                    &mut data,
                    current_pos,
                    &op.post_slice,
                    post_offset
                ));
            }
        }

        return Ok(tvec!(data.into_arc_tensor()));
    }
}

unsafe fn overwrite_part_of_pulse<T: Datum>(
    axis: usize,
    pulse_data: &mut Tensor,
    current_pos: usize,
    const_data: &Tensor,
    const_offset: usize,
) {
    let pulse = pulse_data.shape()[axis];
    let const_length = const_data.shape()[axis];
    let const_range = const_offset..const_offset + const_length;
    let pulse_range = current_pos..current_pos + pulse;
    let axis = Axis(axis);
    let mut pulse_data = pulse_data.to_array_view_mut_unchecked::<T>();
    let const_data = const_data.to_array_view_unchecked::<T>();

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
