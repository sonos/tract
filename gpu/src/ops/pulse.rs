#![allow(unpredictable_function_pointer_comparisons)]
use crate::device::{DeviceContext, get_context};
use crate::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use crate::turn_handler::make_tensor_for_node;
use crate::utils::compute_broadcast_strides;
use std::ops::Range;
use tract_core::internal::*;
use tract_core::ops::array::PadMode;
use tract_pulse_opl::ops::{AffineChunkTrim, Delay, PulsePad};

// ─── GpuDelay ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuDelay {
    pub inner: Delay,
}

impl GpuDelay {
    pub fn new(inner: &Delay) -> Self {
        Self { inner: inner.clone() }
    }
}

impl Op for GpuDelay {
    fn name(&self) -> StaticName {
        "GpuDelay".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.inner.info()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuDelay {
    not_out_of_plan!();

    fn state(&self, ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(GpuDelayState {
            node_id: ctx.node_id,
            buffer: None,
            shift_scratch: None,
            lanes: 0,
        })))
    }
}

impl TypedOp for GpuDelay {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.inner.output_facts(facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        crate::utils::get_device_facts(inputs, |facts| self.inner.cost(facts))
    }

    as_op!();
}

/// Copy `len` steps along `axis` from `src` into `dst`, addressing one lane of
/// each: axis 0 is the lane axis, or the copy spans it when the lanes are absent
/// and the tensors carry a single stream's state.
#[allow(clippy::too_many_arguments)]
fn copy_lane(
    ctx: &dyn DeviceContext,
    dst: &DeviceTensor,
    dst_lane: Option<usize>,
    dst_start: usize,
    src: &DeviceTensor,
    src_lane: Option<usize>,
    src_start: usize,
    axis: usize,
    len: usize,
) -> TractResult<()> {
    let mut zone: TVec<usize> = src.shape().into();
    zone[axis] = len;
    let mut dst_origin = tvec!(0; dst.rank());
    let mut src_origin = tvec!(0; src.rank());
    if let (Some(dst_lane), Some(src_lane)) = (dst_lane, src_lane) {
        zone[0] = 1;
        dst_origin[0] = dst_lane;
        src_origin[0] = src_lane;
    }
    dst_origin[axis] = dst_start;
    src_origin[axis] = src_start;
    ctx.copy_with_origins(&zone, dst, &dst_origin, dst.strides(), src, &src_origin, src.strides())
}

/// Zero each of `lanes` whole, or the whole tensor when it holds a single
/// stream's state and has no lane axis.
fn zero_lanes(
    ctx: &dyn DeviceContext,
    dst: &DeviceTensor,
    lanes: &[LaneId],
    laned: bool,
) -> TractResult<()> {
    let zero = Tensor::zero_dt(dst.datum_type(), &[])?.into_device()?;
    let flat: TVec<usize> = tvec!(0; dst.rank());
    let broadcast: TVec<isize> = tvec!(0; dst.rank());
    let mut zone: TVec<usize> = dst.shape().into();
    if laned {
        zone[0] = 1;
    }
    for lane in lanes {
        let mut origin = tvec!(0; dst.rank());
        if laned {
            origin[0] = lane.0;
        }
        ctx.copy_with_origins(&zone, dst, &origin, dst.strides(), &zero, &flat, &broadcast)?;
    }
    Ok(())
}

/// Broadcast `value` over `range` along `axis` of one lane of `dst`.
fn fill_lane(
    ctx: &dyn DeviceContext,
    dst: &DeviceTensor,
    lane: Option<usize>,
    axis: usize,
    range: Range<usize>,
    value: &DeviceTensor,
) -> TractResult<()> {
    let mut zone: TVec<usize> = dst.shape().into();
    zone[axis] = range.len();
    let mut origin = tvec!(0; dst.rank());
    if let Some(lane) = lane {
        zone[0] = 1;
        origin[0] = lane;
    }
    origin[axis] = range.start;
    ctx.copy_with_origins(
        &zone,
        dst,
        &origin,
        dst.strides(),
        value,
        &tvec!(0; dst.rank()),
        &tvec!(0; dst.rank()),
    )
}

/// The streaming context preceding the current pulse. `lanes` is the extent of
/// the buffer's lane axis, 1 when the state serves a single stream and the
/// buffer has no lane axis at all.
#[derive(Debug, Clone)]
pub struct GpuDelayState {
    pub node_id: usize,
    pub buffer: Option<DeviceTensor>,
    pub shift_scratch: Option<DeviceTensor>,
    lanes: usize,
}

impl GpuDelayState {
    /// One seat's delay: `seat` indexes the batch axis of `input` and `output`,
    /// `lane` the lane axis of the buffer. Both are absent when the state serves
    /// a single stream.
    fn delay_seat(
        &mut self,
        ctx: &dyn DeviceContext,
        op: &Delay,
        input: &DeviceTensor,
        output: &mut DeviceTensor,
        seat: Option<usize>,
        lane: Option<usize>,
    ) -> TractResult<()> {
        let axis = op.axis;
        let buffered = op.delay + op.overlap;
        let input_pulse = input.shape()[axis];
        let output_pulse = input_pulse + op.overlap;
        let from_input = input_pulse.saturating_sub(op.delay);
        let from_buffer = output_pulse.saturating_sub(from_input);
        let buffer = self.buffer.as_ref().unwrap();

        copy_lane(ctx, output, seat, 0, buffer, lane, 0, axis, from_buffer)?;
        copy_lane(ctx, output, seat, from_buffer, input, seat, 0, axis, from_input)?;

        if buffered < input_pulse {
            copy_lane(ctx, buffer, lane, 0, input, seat, input_pulse - buffered, axis, buffered)?;
        } else {
            // CUDA memcpy is undefined for overlapping regions in the same
            // buffer (parallel threads), so shift the lane left by input_pulse
            // through a scratch buffer.
            let keep = buffered - input_pulse;
            let scratch = match self.shift_scratch.as_ref() {
                Some(scratch) => scratch,
                None => {
                    let mut shape: TVec<usize> = buffer.shape().into();
                    if lane.is_some() {
                        shape[0] = 1;
                    }
                    self.shift_scratch
                        .insert(DeviceTensor::uninitialized_dt(input.datum_type(), &shape)?)
                }
            };
            let scratch_lane = lane.map(|_| 0);
            copy_lane(ctx, scratch, scratch_lane, 0, buffer, lane, input_pulse, axis, keep)?;
            copy_lane(ctx, buffer, lane, 0, scratch, scratch_lane, 0, axis, keep)?;
            copy_lane(ctx, buffer, lane, keep, input, seat, 0, axis, input_pulse)?;
        }
        Ok(())
    }
}

impl OpState for GpuDelayState {
    fn eval(
        &mut self,
        ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let op = &op.downcast_ref::<GpuDelay>().ok_or_else(|| format_err!("Wrong Op type"))?.inner;
        let device_input = input.as_device_tensor().context("Expected a GPU tensor")?;
        let mut output_shape: TVec<usize> = device_input.shape().into();
        output_shape[op.axis] = device_input.shape()[op.axis] + op.overlap;
        let dt = device_input.datum_type();
        let device = get_context()?;
        let max_lanes = ctx.seating.max_lanes();
        if self.buffer.is_none() {
            let mut shape: TVec<usize> = device_input.shape().into();
            shape[op.axis] = op.delay + op.overlap;
            if max_lanes > 1 {
                ensure!(op.axis > 0, "GpuDelay on axis 0 leaves no axis 0 for the lanes");
                shape[0] = max_lanes;
            }
            self.buffer = Some(Tensor::zero_dt(dt, &shape)?.into_device()?);
            self.lanes = max_lanes;
        }
        ensure!(
            self.lanes == max_lanes,
            "GpuDelay buffer holds {} lanes, this turn seats {max_lanes} of them",
            self.lanes
        );
        let mut output = make_tensor_for_node(ctx, dt, &output_shape)?;
        if max_lanes == 1 {
            self.delay_seat(&*device, op, device_input, &mut output, None, None)?;
        } else {
            ensure!(
                device_input.shape()[0] == ctx.seating.occupancy(),
                "GpuDelay input carries {} streams, this turn seats {}",
                device_input.shape()[0],
                ctx.seating.occupancy()
            );
            for (seat, lane) in ctx.seating.lanes().iter().enumerate() {
                self.delay_seat(&*device, op, device_input, &mut output, Some(seat), Some(lane.0))?;
            }
        }
        Ok(tvec!(output.into_tensor().into()))
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        let Some(buffer) = self.buffer.as_ref() else { return Ok(()) };
        ensure!(
            lanes.iter().all(|l| l.0 < self.lanes),
            "GpuDelay buffer holds {} lanes, asked to reset {lanes:?}",
            self.lanes
        );
        zero_lanes(&*get_context()?, buffer, lanes, self.lanes > 1)
    }
}

// ─── GpuPulsePad ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPulsePad {
    pub op: PulsePad,
    pub device_cst: Option<DeviceTensor>,
}

impl GpuPulsePad {
    pub fn new(op: &PulsePad) -> TractResult<Self> {
        let device_cst =
            if let PadMode::Constant(c) = &op.mode { Some(c.clone().into_device()?) } else { None };
        Ok(Self { op: op.clone(), device_cst })
    }
}

impl std::hash::Hash for GpuPulsePad {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.op.hash(state);
    }
}

impl Op for GpuPulsePad {
    fn name(&self) -> StaticName {
        "GpuPulsePad".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.op.info()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuPulsePad {
    not_out_of_plan!();

    fn state(&self, ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(GpuPulsePadState {
            node_id: ctx.node_id,
            current_pos: tvec!(),
            last_valid_frame: None,
            lanes: 0,
        })))
    }
}

impl TypedOp for GpuPulsePad {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.op.output_facts(facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        crate::utils::get_device_facts(inputs, |facts| self.op.cost(facts))
    }

    as_op!();
}

/// One padding state per lane: `current_pos` is each lane's position in its own
/// stream and `last_valid_frame` each lane's last frame of valid input, for edge
/// padding. `lanes` is the extent of their lane axis, 1 when the state serves a
/// single stream and they have no lane axis at all.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct GpuPulsePadState {
    node_id: usize,
    current_pos: TVec<usize>,
    last_valid_frame: Option<DeviceTensor>,
    lanes: usize,
}

/// Repeat one frame of `src` over `range` along `axis` of one lane of `dst`.
/// The frame index has to go into the source offset rather than an origin: the
/// axis stride is zeroed to broadcast the frame, and an origin term on that axis
/// is scaled by that same stride.
#[allow(clippy::too_many_arguments)]
fn repeat_frame(
    ctx: &dyn DeviceContext,
    dst: &DeviceTensor,
    dst_lane: Option<usize>,
    axis: usize,
    range: Range<usize>,
    src: &DeviceTensor,
    src_lane: Option<usize>,
    frame: usize,
) -> TractResult<()> {
    let mut zone: TVec<usize> = dst.shape().into();
    zone[axis] = range.len();
    let mut dst_offset = range.start * dst.strides()[axis] as usize;
    let mut src_offset = frame * src.strides()[axis] as usize;
    if let (Some(dst_lane), Some(src_lane)) = (dst_lane, src_lane) {
        zone[0] = 1;
        dst_offset += dst_lane * dst.strides()[0] as usize;
        src_offset += src_lane * src.strides()[0] as usize;
    }
    if zone.iter().product::<usize>() == 0 {
        return Ok(());
    }
    let size = dst.datum_type().size_of();
    let mut src_strides: TVec<isize> = src.strides().into();
    src_strides[axis] = 0;
    ctx.copy_nd(src, src_offset * size, &src_strides, dst, dst_offset * size, &zone, dst.strides())
}

impl GpuPulsePadState {
    fn save_frame(
        &mut self,
        ctx: &dyn DeviceContext,
        op: &PulsePad,
        input: &DeviceTensor,
        frame: usize,
        seat: Option<usize>,
        lane: Option<usize>,
    ) -> TractResult<()> {
        let frames = match self.last_valid_frame.as_ref() {
            Some(frames) => frames,
            None => {
                let mut shape: TVec<usize> = input.shape().into();
                shape[op.axis] = 1;
                if lane.is_some() {
                    shape[0] = self.lanes;
                }
                let frames = Tensor::zero_dt(input.datum_type(), &shape)?.into_device()?;
                self.last_valid_frame.insert(frames)
            }
        };
        copy_lane(ctx, frames, lane, 0, input, seat, frame, op.axis, 1)
    }

    fn pad(
        &mut self,
        ctx: &EvalContext,
        gpu_op: &GpuPulsePad,
        input: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let device = get_context()?;
        let op = &gpu_op.op;
        let pulse = input.shape()[op.axis];
        let end_input = op.end_input.eval(ctx.symbols).to_usize().unwrap_or(usize::MAX);
        let after = op.after.eval(ctx.symbols).to_usize().unwrap_or(usize::MAX);
        let max_lanes = ctx.seating.max_lanes();
        if self.lanes == 0 {
            self.lanes = max_lanes;
            self.current_pos = tvec!(0; max_lanes);
        }
        ensure!(
            self.lanes == max_lanes,
            "GpuPulsePad holds {} lanes, this turn seats {max_lanes} of them",
            self.lanes
        );
        let occupancy = if max_lanes == 1 { 1 } else { ctx.seating.occupancy() };
        if max_lanes > 1 {
            ensure!(op.axis > 0, "GpuPulsePad on axis 0 leaves no axis 0 for the lanes");
            ensure!(
                input.shape()[0] == occupancy,
                "GpuPulsePad input carries {} streams, this turn seats {occupancy}",
                input.shape()[0]
            );
        }
        // Seats whose pulse is neither entirely valid input nor entirely outside
        // it, with the stream position they start at. Every other seat is copied
        // over unchanged.
        let mut to_pad: TVec<(Option<usize>, Option<usize>, usize)> = tvec!();
        for ix in 0..occupancy {
            let (seat, lane) = if max_lanes == 1 {
                (None, None)
            } else {
                (Some(ix), Some(ctx.seating.lanes()[ix].0))
            };
            let pulse_begin = self.current_pos[lane.unwrap_or(0)];
            let pulse_end = pulse_begin + pulse;
            self.current_pos[lane.unwrap_or(0)] += pulse - op.overlap;
            if let PadMode::Edge = op.mode
                && after != 0
                && pulse_begin < end_input
            {
                let latest_valid_frame = (end_input - pulse_begin).min(pulse) - 1;
                self.save_frame(&*device, op, input, latest_valid_frame, seat, lane)?;
            }
            let valid = pulse_begin >= op.begin_input && pulse_end <= end_input;
            let outside = pulse_end <= op.begin_input - op.before
                || pulse_begin >= end_input.saturating_add(after);
            if !valid && !outside {
                to_pad.push((seat, lane, pulse_begin));
            }
        }

        // Start with a copy of input.  The fused-axis-op chain may have
        // installed a non-contiguous view (Move only permutes strides,
        // never materialises), so a flat memcpy would read the buffer in
        // pre-Move order; copy_nd honours `input.strides()` instead.
        let output = make_tensor_for_node(ctx, input.datum_type(), input.shape())?;
        device.copy_nd(input, 0, input.strides(), &output, 0, input.shape(), output.strides())?;

        for (seat, lane, pulse_begin) in to_pad {
            if pulse_begin < op.begin_input {
                let fill_up_to = (op.begin_input - pulse_begin).min(pulse);
                match &op.mode {
                    PadMode::Constant(_) => fill_lane(
                        &*device,
                        &output,
                        seat,
                        op.axis,
                        0..fill_up_to,
                        gpu_op.device_cst.as_ref().unwrap(),
                    )?,
                    PadMode::Edge => repeat_frame(
                        &*device,
                        &output,
                        seat,
                        op.axis,
                        0..fill_up_to,
                        input,
                        seat,
                        fill_up_to,
                    )?,
                    _ => unimplemented!(),
                }
            }

            if pulse_begin + pulse > end_input && after > 0 {
                let fill_from = pulse - (pulse_begin + pulse - end_input).min(pulse);
                match &op.mode {
                    PadMode::Constant(_) => fill_lane(
                        &*device,
                        &output,
                        seat,
                        op.axis,
                        fill_from..pulse,
                        gpu_op.device_cst.as_ref().unwrap(),
                    )?,
                    PadMode::Edge => repeat_frame(
                        &*device,
                        &output,
                        seat,
                        op.axis,
                        fill_from..pulse,
                        self.last_valid_frame.as_ref().unwrap(),
                        lane,
                        0,
                    )?,
                    _ => unimplemented!(),
                }
            }
        }
        Ok(output)
    }
}

impl OpState for GpuPulsePadState {
    fn eval(
        &mut self,
        ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let gpu_op =
            op.downcast_ref::<GpuPulsePad>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let device_input = input.as_device_tensor().context("Expected a GPU tensor")?;
        let output = self.pad(ctx, gpu_op, device_input)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        if self.lanes == 0 {
            return Ok(());
        }
        ensure!(
            lanes.iter().all(|l| l.0 < self.lanes),
            "GpuPulsePad holds {} lanes, asked to reset {lanes:?}",
            self.lanes
        );
        for lane in lanes {
            self.current_pos[lane.0] = 0;
        }
        if let Some(frames) = self.last_valid_frame.as_ref() {
            zero_lanes(&*get_context()?, frames, lanes, self.lanes > 1)?;
        }
        Ok(())
    }
}

// ─── GpuAffineChunkTrim ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuAffineChunkTrim {
    pub inner: AffineChunkTrim,
}

impl GpuAffineChunkTrim {
    pub fn new(inner: &AffineChunkTrim) -> Self {
        Self { inner: inner.clone() }
    }
}

impl Op for GpuAffineChunkTrim {
    fn name(&self) -> StaticName {
        "GpuAffineChunkTrim".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.inner.info()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuAffineChunkTrim {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let axis = self.inner.axis;
        let n = input.shape()[axis];
        let take = if n.saturating_sub(self.inner.typed_trim) >= self.inner.target_per_pulse {
            n - self.inner.typed_trim
        } else {
            n
        };
        if take == n {
            return Ok(tvec!(input_value));
        }
        let mut o_shape: TVec<usize> = input.shape().into();
        o_shape[axis] = take;
        let output = make_tensor_for_node(ctx, input.datum_type(), &o_shape)?;
        let broadcast_strides = compute_broadcast_strides(&o_shape, input.strides())?;
        let device = get_context()?;
        device.copy_nd(
            input,
            0,
            &broadcast_strides,
            &output,
            0,
            output.shape(),
            output.strides(),
        )?;
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuAffineChunkTrim {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.inner.output_facts(facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}
