#![allow(unpredictable_function_pointer_comparisons)]
use crate::device::{DeviceContext, get_context};
use crate::session_handler::make_tensor_for_node;
use crate::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use std::ops::Range;
use tract_core::internal::*;
use tract_core::ops::array::PadMode;
use tract_core::trivial_op_state_freeze;
use tract_pulse_opl::ops::{Delay, PulsePad};

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
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(&self, _session: &TurnState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(GpuDelayState { node_id, buffer: None })))
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

#[derive(Debug, Clone)]
pub struct GpuDelayState {
    pub node_id: usize,
    pub buffer: Option<DeviceTensor>,
}

impl GpuDelayState {
    unsafe fn apply_delay_unchecked(
        &mut self,
        ctx: &dyn DeviceContext,
        op: &Delay,
        input: &DeviceTensor,
        output: &mut DeviceTensor,
    ) -> TractResult<()> {
        let buffered = op.delay + op.overlap;
        let input_pulse = input.shape()[op.axis];
        let output_pulse = input_pulse + op.overlap;
        let buffer = self.buffer.as_mut().unwrap();

        let from_input = input_pulse.saturating_sub(op.delay);
        let from_buffer = output_pulse.saturating_sub(from_input);

        // Copy from buffer to output
        ctx.assign_slice(output, 0..from_buffer, buffer, 0..from_buffer, op.axis)?;
        // Copy from input to output
        ctx.assign_slice(output, from_buffer..output_pulse, input, 0..from_input, op.axis)?;

        // Maintain buffer
        if buffered < input_pulse {
            ctx.assign_slice(
                buffer,
                0..buffered,
                input,
                (input_pulse - buffered)..input_pulse,
                op.axis,
            )?;
        } else {
            // Shift buffer left by input_pulse elements.
            // CUDA memcpy is undefined for overlapping regions in the same
            // buffer, so copy via a temporary.
            let keep = buffered - input_pulse;
            let temp = DeviceTensor::uninitialized_dt(input.datum_type(), buffer.shape())?;
            ctx.assign_slice(&temp, 0..keep, buffer, input_pulse..buffered, op.axis)?;
            ctx.assign_slice(buffer, 0..keep, &temp, 0..keep, op.axis)?;
            // Copy input to end of buffer
            ctx.assign_slice(
                buffer,
                (buffered - input_pulse)..buffered,
                input,
                0..input_pulse,
                op.axis,
            )?;
        }
        Ok(())
    }
}

impl OpState for GpuDelayState {
    fn eval(
        &mut self,
        state: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let op = &op.downcast_ref::<GpuDelay>().ok_or_else(|| format_err!("Wrong Op type"))?.inner;
        let buffered = op.delay + op.overlap;
        let device_input = input.as_device_tensor().context("Expected a GPU tensor")?;
        let input_pulse = device_input.shape()[op.axis];
        let output_pulse = input_pulse + op.overlap;
        let mut output_shape: TVec<usize> = device_input.shape().into();
        output_shape[op.axis] = output_pulse;
        let dt = device_input.datum_type();
        let ctx = get_context()?;
        unsafe {
            if self.buffer.is_none() {
                let mut shape = device_input.shape().to_owned();
                shape[op.axis] = buffered;
                self.buffer = Some(Tensor::zero_dt(dt, &shape)?.into_device()?);
            };
            let mut output = make_tensor_for_node(state, self.node_id, dt, &output_shape)?;
            self.apply_delay_unchecked(&*ctx, op, device_input, &mut output)?;
            Ok(tvec!(output.into_tensor().into()))
        }
    }
}

trivial_op_state_freeze!(GpuDelayState);

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
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(&self, _session: &TurnState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(GpuPulsePadState { node_id, current_pos: 0, last_valid_frame: None })))
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

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct GpuPulsePadState {
    node_id: usize,
    current_pos: usize,
    last_valid_frame: Option<DeviceTensor>,
}

fn fill_slice_constant(
    ctx: &dyn DeviceContext,
    dst: &mut DeviceTensor,
    cst: &DeviceTensor,
    axis: usize,
    range: Range<usize>,
) -> TractResult<()> {
    let mut zone_shape: TVec<usize> = dst.shape().into();
    zone_shape[axis] = range.len();
    let mut dst_origin = tvec!(0; dst.rank());
    dst_origin[axis] = range.start;
    ctx.copy_with_origins(
        &zone_shape,
        dst,
        &dst_origin,
        dst.strides(),
        cst,
        &tvec!(0; dst.rank()),
        &tvec!(0; dst.rank()),
    )
}

fn fill_slice_repeating_one_frame(
    ctx: &dyn DeviceContext,
    dst: &mut DeviceTensor,
    src: &DeviceTensor,
    axis: usize,
    dst_range: Range<usize>,
    src_frame: usize,
) -> TractResult<()> {
    let mut zone_shape: TVec<usize> = dst.shape().into();
    zone_shape[axis] = dst_range.len();
    let mut dst_origin = tvec!(0; dst.rank());
    dst_origin[axis] = dst_range.start;
    let mut src_origin = tvec!(0; src.rank());
    src_origin[axis] = src_frame;
    let mut src_strides: TVec<isize> = src.strides().into();
    src_strides[axis] = 0;
    ctx.copy_with_origins(
        &zone_shape,
        dst,
        &dst_origin,
        dst.strides(),
        src,
        &src_origin,
        &src_strides,
    )
}

impl GpuPulsePadState {
    fn save_frame(
        &mut self,
        ctx: &dyn DeviceContext,
        op: &PulsePad,
        input: &DeviceTensor,
        frame: usize,
    ) -> TractResult<()> {
        let mut frame_shape: TVec<usize> = input.shape().into();
        frame_shape[op.axis] = 1;
        let last_valid_frame = DeviceTensor::uninitialized_dt(input.datum_type(), &frame_shape)?;
        ctx.assign_slice(&last_valid_frame, 0..1, input, frame..frame + 1, op.axis)?;
        self.last_valid_frame = Some(last_valid_frame);
        Ok(())
    }

    fn pad(
        &mut self,
        session: &TurnState,
        gpu_op: &GpuPulsePad,
        input: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let ctx = get_context()?;
        let op = &gpu_op.op;
        let pulse = input.shape()[op.axis];
        let pulse_begin = self.current_pos;
        let pulse_end = self.current_pos + pulse;
        self.current_pos += pulse - op.overlap;
        let end_input =
            op.end_input.eval(&session.resolved_symbols).to_usize().unwrap_or(usize::MAX);
        let after = op.after.eval(&session.resolved_symbols).to_usize().unwrap_or(usize::MAX);

        if let PadMode::Edge = op.mode
            && after != 0
            && pulse_begin < end_input
        {
            let latest_valid_frame = (end_input - pulse_begin).min(pulse) - 1;
            self.save_frame(&*ctx, op, input, latest_valid_frame)?;
        }

        // Start with a copy of input
        let mut output =
            make_tensor_for_node(session, self.node_id, input.datum_type(), input.shape())?;
        let flat_len = input.len() * input.datum_type().size_of();
        ctx.flat_copy(input, 0, &output, 0, flat_len)?;

        // Quick return if entirely in valid or invalid range
        if (pulse_begin >= op.begin_input && pulse_end <= end_input)
            || (pulse_end <= op.begin_input - op.before
                || pulse_begin >= end_input.saturating_add(after))
        {
            return Ok(output);
        }

        if pulse_begin < op.begin_input {
            let fill_up_to = (op.begin_input - pulse_begin).min(pulse);
            match &op.mode {
                PadMode::Constant(_) => fill_slice_constant(
                    &*ctx,
                    &mut output,
                    gpu_op.device_cst.as_ref().unwrap(),
                    op.axis,
                    0..fill_up_to,
                )?,
                PadMode::Edge => fill_slice_repeating_one_frame(
                    &*ctx,
                    &mut output,
                    input,
                    op.axis,
                    0..fill_up_to,
                    fill_up_to,
                )?,
                _ => unimplemented!(),
            }
        }

        if pulse_end > end_input {
            let fill_from = pulse - (pulse_end - end_input).min(pulse);
            match &op.mode {
                PadMode::Constant(_) => fill_slice_constant(
                    &*ctx,
                    &mut output,
                    gpu_op.device_cst.as_ref().unwrap(),
                    op.axis,
                    fill_from..pulse,
                )?,
                PadMode::Edge => fill_slice_repeating_one_frame(
                    &*ctx,
                    &mut output,
                    self.last_valid_frame.as_ref().unwrap(),
                    op.axis,
                    fill_from..pulse,
                    0,
                )?,
                _ => unimplemented!(),
            }
        }
        Ok(output)
    }
}

impl OpState for GpuPulsePadState {
    fn eval(
        &mut self,
        session: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let gpu_op =
            op.downcast_ref::<GpuPulsePad>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let device_input = input.as_device_tensor().context("Expected a GPU tensor")?;
        let output = self.pad(session, gpu_op, device_input)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

trivial_op_state_freeze!(GpuPulsePadState);
