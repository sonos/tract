use std::ops::Range;
use tract_nnef::internal::*;

/// Concat with pulse along concat axis
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PulsedSameAxisConcat {
    pub axis: usize,
    pub before_len: usize,
    pub after_len: usize,
    pub input_delay: usize,
    pub input_len: TDim,
}

impl Op for PulsedSameAxisConcat {
    fn name(&self) -> StaticName {
        "PulsedSameAxisConcat".into()
    }

    op_as_typed_op!();
}

impl EvalOp for PulsedSameAxisConcat {
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<PulsedSameAxisConcatState>::default()))
    }
}

impl TypedOp for PulsedSameAxisConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        // Inputs are (pre_prefix, stream, post_suffix); the streaming output
        // has the same shape as `stream` (input 1).  The pulsed-op version
        // builds its PulsedFact from the same input.  Previously this
        // returned `inputs[0]` — the small constant pre-buffer — which broke
        // any pass that re-derived the typed shape post-pulsification (e.g.
        // CUDA/Metal translation walking the pulsified preprocessor: every
        // downstream op saw the pre-buffer shape instead of the pulse-axis
        // size and produced collapsed outputs).
        ensure!(inputs.len() == 3, "Expect 3 inputs");
        Ok(tvec!(inputs[1].clone()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct PulsedSameAxisConcatState {
    /// How far along the stream each lane stands, in frames of the pulse axis:
    /// what says whether the pulse now arriving overlaps the leading context or
    /// the trailing tail. One entry per lane, since a turn seats a position of
    /// each of the streams it carries. Empty until the first turn sizes it.
    positions: TVec<usize>,
}

impl OpState for PulsedSameAxisConcatState {
    fn eval(
        &mut self,
        ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op
            .downcast_ref::<PulsedSameAxisConcat>()
            .ok_or_else(|| format_err!("Wrong Op type"))?;
        let (pre, input, post) = args_3!(inputs);
        let mut data = input.into_tensor();
        let pulse = data.shape()[op.axis];
        let max_lanes = ctx.seating.max_lanes();
        if self.positions.is_empty() {
            ensure!(
                max_lanes == 1 || op.axis > 0,
                "PulsedSameAxisConcat on axis 0 leaves no axis 0 for the lanes"
            );
            self.positions = tvec!(0; max_lanes);
        }
        ensure!(
            self.positions.len() == max_lanes,
            "PulsedSameAxisConcat stands in {} lanes, this turn seats {max_lanes} of them",
            self.positions.len()
        );
        if max_lanes > 1 {
            ensure!(
                data.shape()[0] == ctx.seating.occupancy(),
                "PulsedSameAxisConcat input carries {} streams, this turn seats {}",
                data.shape()[0],
                ctx.seating.occupancy()
            );
        }

        let pre_length = pre.shape()[op.axis];
        let pre_offset = op.input_delay - pre_length;
        let post_offset =
            op.input_len.maybe_eval_to_i64(ctx.symbols).map(|l| op.input_delay + l as usize);
        for ix in 0..ctx.seating.occupancy() {
            let (seat, lane) = ctx.seating.address(ix);
            let position = self.positions[lane.unwrap_or(0)];
            overwrite_part_of_pulse(op.axis, &mut data, position, &pre, pre_offset, seat)?;
            if let Some(post_offset) = post_offset {
                overwrite_part_of_pulse(op.axis, &mut data, position, &post, post_offset, seat)?;
            }
            self.positions[lane.unwrap_or(0)] = position + pulse;
        }

        Ok(tvec!(data.into_tvalue()))
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        if self.positions.is_empty() {
            return Ok(());
        }
        ensure!(
            lanes.iter().all(|l| l.0 < self.positions.len()),
            "PulsedSameAxisConcat stands in {} lanes, asked to reset {lanes:?}",
            self.positions.len()
        );
        for lane in lanes {
            self.positions[lane.0] = 0;
        }
        Ok(())
    }
}

/// Overwrite, in the seat's own sub-tensor, whatever of `pulse_data` the
/// constant part covers: `const_data` sits at `const_offset` in the stream, the
/// pulse at `current_pos`, and only their overlap is assigned.
pub fn overwrite_part_of_pulse(
    axis: usize,
    pulse_data: &mut Tensor,
    current_pos: usize,
    const_data: &Tensor,
    const_offset: usize,
    seat: Option<usize>,
) -> TractResult<()> {
    let pulse = pulse_data.shape()[axis];
    let const_length = const_data.shape()[axis];
    let const_range = const_offset..const_offset + const_length;
    let pulse_range = current_pos..current_pos + pulse;
    let assign = |pulse_data: &mut Tensor, range: Range<usize>, src_range: Range<usize>| {
        pulse_data.assign_slice_at_prefix(
            seat.as_slice(),
            range,
            const_data,
            seat.as_slice(),
            src_range,
            axis,
        )
    };

    match range_in_range(&pulse_range, &const_range) {
        RangeInRange::Before(_) | RangeInRange::After(_) => (),
        RangeInRange::Begin(offset) => {
            // ----[<----->HHH]HH----
            assign(pulse_data, offset..pulse, 0..pulse - offset)?;
        }
        RangeInRange::Contain(offset) => {
            // ----[<----->HHHHHHH-]---
            assign(pulse_data, offset..offset + const_length, 0..const_length)?;
        }
        RangeInRange::Inside(offset) => {
            // ----------<H>[HH]HH----
            assign(pulse_data, 0..pulse, offset..offset + pulse)?;
        }
        RangeInRange::End(offset) => {
            // --------<HHH>[HHHH-]---
            assign(pulse_data, 0..const_length - offset, offset..const_length)?;
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
