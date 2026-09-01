use tract_nnef::internal::*;

use crate::lane::lane_runs;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_pulse_delay",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("delay"),
            TypeName::Integer.named("overlap"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_delay,
    );
}

fn de_delay(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as::<i64>(builder, "axis")? as usize;
    let delay = invocation.named_arg_as::<i64>(builder, "delay")? as usize;
    let overlap = invocation.named_arg_as::<i64>(builder, "overlap")? as usize;
    let input_fact = builder.model.outlet_fact(wire)?;
    let op = Delay::new_typed(input_fact, axis, delay, overlap);
    builder.wire(op, &[wire])
}

/// The streaming context preceding the current pulse. `lanes` is the extent of
/// the buffer's lane axis, 1 when the state serves a single stream and the
/// buffer has no lane axis at all.
#[derive(Debug, Clone, Default)]
pub struct DelayState {
    pub buffer: Option<Tensor>,
    lanes: usize,
}

impl DelayState {
    /// One seat's delay: `seat` indexes the batch axis of `input` and `output`,
    /// `lane` the lane axis of the buffer. Both are absent when the state serves
    /// a single stream.
    fn delay_seat(
        &mut self,
        op: &Delay,
        input: &Tensor,
        output: &mut Tensor,
        seat: Option<usize>,
        lane: Option<usize>,
    ) -> TractResult<()> {
        let axis = op.axis;
        let buffered = op.delay + op.overlap;
        let input_pulse = input.shape()[axis];
        let output_pulse = input_pulse + op.overlap;
        let from_input = input_pulse.saturating_sub(op.delay);
        let from_buffer = output_pulse.saturating_sub(from_input);
        let buffer = self.buffer.as_mut().unwrap();
        output.assign_slice_at_prefix(
            seat.as_slice(),
            0..from_buffer,
            buffer,
            lane.as_slice(),
            0..from_buffer,
            axis,
        )?;
        output.assign_slice_at_prefix(
            seat.as_slice(),
            from_buffer..output_pulse,
            input,
            seat.as_slice(),
            0..from_input,
            axis,
        )?;
        if buffered < input_pulse {
            let tail = input_pulse - buffered;
            buffer.assign_slice_at_prefix(
                lane.as_slice(),
                0..buffered,
                input,
                seat.as_slice(),
                tail..input_pulse,
                axis,
            )?;
        } else {
            let keep = buffered - input_pulse;
            // The kept context moves down inside the buffer, so source and
            // destination are the same tensor and no assign can name both.
            let dt_size = buffer.datum_type().size_of();
            let bshape: TVec<usize> = buffer.shape().into();
            let source = lane_runs(&bshape, dt_size, axis, lane, input_pulse..buffered);
            let buf = buffer.as_bytes_mut();
            for (to, from) in lane_runs(&bshape, dt_size, axis, lane, 0..keep).zip(source) {
                buf.copy_within(from, to.start);
            }
            buffer.assign_slice_at_prefix(
                lane.as_slice(),
                keep..buffered,
                input,
                seat.as_slice(),
                0..input_pulse,
                axis,
            )?;
        }
        Ok(())
    }
}

impl OpState for DelayState {
    fn eval(
        &mut self,
        ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<Delay>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let dt = input.datum_type();
        ensure!(dt.is_copy(), "Delay buffers {dt:?}, which is not copy");
        let max_lanes = ctx.seating.max_lanes();
        let mut output_shape: TVec<usize> = input.shape().into();
        output_shape[op.axis] = input.shape()[op.axis] + op.overlap;
        if self.buffer.is_none() {
            let mut shape: TVec<usize> = input.shape().into();
            shape[op.axis] = op.delay + op.overlap;
            if max_lanes > 1 {
                ensure!(op.axis > 0, "Delay on axis 0 leaves no axis 0 for the lanes");
                shape[0] = max_lanes;
            }
            // Zero-init: the buffer holds the streaming context preceding the
            // first pulse, and silence (zero) is the only sensible default.
            // Uninitialized memory leaks into the first `delay` output frames
            // and diverges from the GPU op (which zero-inits), making any
            // per-node comparison meaningless on the warmup region.
            self.buffer = Some(Tensor::zero_dt(dt, &shape)?);
            self.lanes = max_lanes;
        }
        ensure!(
            self.lanes == max_lanes,
            "Delay buffer holds {} lanes, this turn seats {max_lanes} of them",
            self.lanes
        );
        let mut output = unsafe { Tensor::uninitialized_dt(dt, &output_shape)? };
        if max_lanes == 1 {
            self.delay_seat(op, &input, &mut output, None, None)?;
        } else {
            ensure!(
                input.shape()[0] == ctx.seating.occupancy(),
                "Delay input carries {} streams, this turn seats {}",
                input.shape()[0],
                ctx.seating.occupancy()
            );
            for (seat, lane) in ctx.seating.lanes().iter().enumerate() {
                self.delay_seat(op, &input, &mut output, Some(seat), Some(lane.0))?;
            }
        }
        Ok(tvec!(output.into()))
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        let Some(buffer) = self.buffer.as_mut() else { return Ok(()) };
        ensure!(
            lanes.iter().all(|l| l.0 < self.lanes),
            "Delay buffer holds {} lanes, asked to reset {lanes:?}",
            self.lanes
        );
        let stride = buffer.as_bytes().len() / self.lanes;
        for lane in lanes {
            buffer.as_bytes_mut()[lane.0 * stride..][..stride].fill(0);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Delay {
    pub buffer_shape: TVec<TDim>,
    pub axis: usize,
    pub delay: usize,
    pub overlap: usize,
}

impl Delay {
    pub fn new_typed(input_fact: &TypedFact, axis: usize, delay: usize, overlap: usize) -> Delay {
        let mut buffer_shape: TVec<TDim> = input_fact.shape.to_tvec();
        buffer_shape[axis] = (delay + overlap).to_dim();
        Delay { buffer_shape, axis, delay, overlap }
    }
}

impl Op for Delay {
    fn name(&self) -> StaticName {
        "Delay".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!("axis: {} delay: {} overlap: {}", self.axis, self.delay, self.overlap),
            format!("buffer: {:?}", self.buffer_shape),
        ])
    }

    op_as_typed_op!();
}

impl EvalOp for Delay {
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(DelayState::default())))
    }
}

impl TypedOp for Delay {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, fact.shape[self.axis].clone() + self.overlap.to_dim());
        Ok(tvec!(fact))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((Cost::Buffer(inputs[0].datum_type), self.buffer_shape.iter().product())))
    }

    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        if self.axis != 0 {
            Ok(tvec!((InOut::In(0), AxisOp::Move(self.axis, 0))))
        } else {
            Ok(tvec!())
        }
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some(axis) = change.transform_axis(self.axis) {
            if axis != self.axis {
                Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(Self { axis, ..self.clone() }) as _),
                    change,
                )))
            } else {
                Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
            }
        } else {
            Ok(None)
        }
    }
}
