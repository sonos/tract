use std::ops::Range;

use tract_nnef::internal::*;

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

/// The streaming context preceding the current pulse, held as a ring: `heads`
/// is each lane's ring index of the oldest buffered frame, so a turn overwrites
/// the frames it retires instead of shifting the whole buffer down. `lanes` is
/// the extent of the buffer's lane axis, 1 when the state serves a single
/// stream and the buffer has no lane axis at all.
#[derive(Debug, Clone, Default)]
pub struct DelayState {
    pub buffer: Option<Tensor>,
    heads: TVec<usize>,
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
        let input_pulse = input.shape()[axis];
        let output_pulse = input_pulse + op.overlap;
        let from_input = input_pulse.saturating_sub(op.delay);
        let from_buffer = output_pulse - from_input;
        let head = self.heads[lane.unwrap_or(0)];
        let buffer = self.buffer.as_mut().unwrap();
        for (at, run) in op.ring_runs(head, from_buffer) {
            let len = run.len();
            output.assign_slice_at_prefix(
                seat.as_slice(),
                at..at + len,
                buffer,
                lane.as_slice(),
                run,
                axis,
            )?;
        }
        output.assign_slice_at_prefix(
            seat.as_slice(),
            from_buffer..output_pulse,
            input,
            seat.as_slice(),
            0..from_input,
            axis,
        )?;
        let fresh = input_pulse.min(op.buffered());
        for (at, run) in op.ring_runs(op.ring_index(head + input_pulse - fresh), fresh) {
            let len = run.len();
            let from = input_pulse - fresh + at;
            buffer.assign_slice_at_prefix(
                lane.as_slice(),
                run,
                input,
                seat.as_slice(),
                from..from + len,
                axis,
            )?;
        }
        self.heads[lane.unwrap_or(0)] = op.ring_index(head + input_pulse);
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
            self.heads = tvec!(0; max_lanes);
        }
        ensure!(
            self.lanes == max_lanes,
            "Delay buffer holds {} lanes, this turn seats {max_lanes} of them",
            self.lanes
        );
        let mut output = unsafe { Tensor::uninitialized_dt(dt, &output_shape)? };
        if max_lanes > 1 {
            ensure!(
                input.shape()[0] == ctx.seating.occupancy(),
                "Delay input carries {} streams, this turn seats {}",
                input.shape()[0],
                ctx.seating.occupancy()
            );
        }
        for ix in 0..ctx.seating.occupancy() {
            let (seat, lane) = ctx.seating.address(ix);
            self.delay_seat(op, &input, &mut output, seat, lane)?;
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
            self.heads[lane.0] = 0;
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

    /// The number of frames the state buffers, and so the length of its ring.
    pub fn buffered(&self) -> usize {
        self.delay + self.overlap
    }

    /// Wrap a ring index which may have run one lap past the end.
    pub fn ring_index(&self, index: usize) -> usize {
        if self.buffered() == 0 { 0 } else { index % self.buffered() }
    }

    /// The move `suggested_axis_changes` asks for, when the layout has one to
    /// gain: the ring copies runs of frames, and a run is contiguous only when
    /// the axis leads. A symbolic leading extent is the batch axis a laned turn
    /// addresses its buffer by, and displacing it would leave the state
    /// un-lane-addressable, so the layout stays as it is.
    fn wants_axis_first(&self) -> bool {
        self.axis != 0 && self.buffer_shape[0].as_i64().is_some()
    }

    /// The one or two runs of the ring holding `len` frames from ring index
    /// `start`, each paired with its offset in the contiguous sequence of
    /// frames they spell out.
    pub fn ring_runs(&self, start: usize, len: usize) -> TVec<(usize, Range<usize>)> {
        if len == 0 {
            return tvec!();
        }
        let first = len.min(self.buffered() - start);
        let mut runs = tvec!((0, start..start + first));
        if first < len {
            runs.push((first, 0..len - first));
        }
        runs
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
        if self.wants_axis_first() {
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
