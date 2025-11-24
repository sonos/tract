use std::ops::Range;

use derive_new::new;
use tract_core::ops::array::PadMode;
use tract_core::{internal::*, trivial_op_state_freeeze};
use tract_gpu::session_handler::make_tensor_for_node;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_pulse_opl::ops::{Delay, PulsePad};

use crate::kernels::{
    get_cuda_view, get_cuda_view_mut, get_sliced_cuda_view, get_sliced_cuda_view_mut,
};
use crate::tensor::{device_tensor_assign_slice, device_tensor_launch_copy};
use crate::CUDA_STREAM;

#[derive(Debug, new, Clone)]
pub struct CudaDelay(Delay);

impl Op for CudaDelay {
    fn name(&self) -> StaticName {
        "CudaDelay".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.0.info()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaDelay {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(CudaDelayState { node_id, buffer: None })))
    }
}

impl TypedOp for CudaDelay {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |input_facts| {
            self.0.output_facts(input_facts)
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |input_facts| self.0.cost(input_facts))
            .with_context(|| format!("Error while computing cost for {:?}", self.name()))
    }

    as_op!();
}

#[derive(Debug, Clone)]
pub struct CudaDelayState {
    pub node_id: usize,
    pub buffer: Option<DeviceTensor>,
}

impl CudaDelayState {
    pub unsafe fn apply_delay_unchecked(
        &mut self,
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
        CUDA_STREAM.with(|stream| -> TractResult<()> {
            device_tensor_assign_slice(
                stream,
                output,
                ..from_buffer,
                buffer,
                ..from_buffer,
                op.axis,
            )?;
            device_tensor_assign_slice(
                stream,
                output,
                from_buffer..,
                input,
                ..from_input,
                op.axis,
            )?;

            // maintain buffer
            if buffered < input_pulse {
                device_tensor_assign_slice(
                    stream,
                    buffer,
                    ..,
                    input,
                    (input_pulse - buffered)..,
                    op.axis,
                )?;
            } else {
                let dt = input.datum_type();
                let offset = buffer.strides()[op.axis] as usize * dt.size_of() * input_pulse;
                let moved = buffer.len() * dt.size_of() - offset;
                let scratch = DeviceTensor::uninitialized_dt(u8::datum_type(), &[moved])?;
                stream.memcpy_dtod(
                    &get_sliced_cuda_view(buffer, offset, moved)?,
                    &mut get_cuda_view_mut(&scratch),
                )?;
                stream.memcpy_dtod(
                    &get_cuda_view(&scratch),
                    &mut get_sliced_cuda_view_mut(buffer, 0, moved)?,
                )?;
                device_tensor_assign_slice(
                    stream,
                    buffer,
                    (buffered - input_pulse)..,
                    input,
                    ..,
                    op.axis,
                )?;
            }
            Ok(())
        })
    }
}

impl OpState for CudaDelayState {
    fn eval(
        &mut self,
        state: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let op = &op.downcast_ref::<CudaDelay>().ok_or_else(|| format_err!("Wrong Op type"))?.0;
        let buffered = op.delay + op.overlap;
        let device_input = input.as_device_tensor().context("Expected a cuda tensor")?;
        let input_pulse = device_input.shape()[op.axis];
        let output_pulse = input_pulse + op.overlap;
        let mut output_shape: TVec<usize> = device_input.shape().into();
        output_shape[op.axis] = output_pulse;
        let dt = device_input.datum_type();
        // build output
        unsafe {
            if self.buffer.is_none() {
                let mut shape = device_input.shape().to_owned();
                shape[op.axis] = buffered;
                self.buffer =
                    Some(DeviceTensor::uninitialized_dt(device_input.datum_type(), &shape)?);
            };
            let mut output = make_tensor_for_node(state, self.node_id, dt, &output_shape)?;
            self.apply_delay_unchecked(op, device_input, &mut output)?;
            Ok(tvec!(output.into_opaque_tensor().into()))
        }
    }
}

trivial_op_state_freeeze!(CudaDelayState);

#[derive(Debug, Clone)]
pub struct CudaPulsePad {
    op: PulsePad,
    device_cst: Option<DeviceTensor>,
}

impl CudaPulsePad {
    pub fn new(op: &PulsePad) -> TractResult<CudaPulsePad> {
        let device_cst =
            if let PadMode::Constant(c) = &op.mode { Some(c.clone().into_device()?) } else { None };
        Ok(CudaPulsePad { op: op.clone(), device_cst })
    }
}

impl Op for CudaPulsePad {
    fn name(&self) -> StaticName {
        "CudaDelay".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.op.info()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaPulsePad {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(CudaPulsePadOpState { node_id, current_pos: 0, last_valid_frame: None })))
    }
}

impl TypedOp for CudaPulsePad {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |input_facts| {
            self.op.output_facts(input_facts)
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |input_facts| self.op.cost(input_facts))
            .with_context(|| format!("Error while computing cost for {:?}", self.name()))
    }

    as_op!();
}
#[derive(Debug, Clone, Hash)]
struct CudaPulsePadOpState {
    node_id: usize,
    current_pos: usize,
    last_valid_frame: Option<DeviceTensor>,
}

impl OpState for CudaPulsePadOpState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs).into_tensor();
        let op = op.downcast_ref::<CudaPulsePad>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let input = input.to_device_tensor()?;
        let tensor = self.pad(session, op, input)?;
        Ok(tvec!(tensor.into_opaque_tensor().into()))
    }
}

fn fill_slice_constant(
    dst: &mut DeviceTensor,
    cst: &DeviceTensor,
    axis: usize,
    range: Range<usize>,
) -> TractResult<()> {
    CUDA_STREAM.with(|stream| {
        let mut zone_shape: TVec<usize> = dst.shape().into();
        zone_shape[axis] = range.len();
        let mut dst_origin = tvec!(0; dst.rank());
        dst_origin[axis] = range.start;
        device_tensor_launch_copy(
            stream,
            &zone_shape,
            dst,
            &dst_origin,
            dst.strides(),
            cst,
            &tvec!(0; dst.rank()),
            &tvec!(0; dst.rank()),
        )
    })
}

fn fill_slice_repeating_one_frame(
    dst: &mut DeviceTensor,
    src: &DeviceTensor,
    axis: usize,
    dst_range: Range<usize>,
    src_frame: usize,
) -> TractResult<()> {
    CUDA_STREAM.with(|stream| {
        let mut zone_shape: TVec<usize> = dst.shape().into();
        zone_shape[axis] = dst_range.len();
        let mut dst_origin = tvec!(0; dst.rank());
        dst_origin[axis] = dst_range.start;
        let mut src_origin = tvec!(0; src.rank());
        src_origin[axis] = src_frame;
        let mut src_strides: TVec<isize> = src.strides().into();
        src_strides[axis] = 0;
        device_tensor_launch_copy(
            stream,
            &zone_shape,
            dst,
            &dst_origin,
            dst.strides(),
            src,
            &src_origin,
            &src_strides,
        )
    })
}

impl CudaPulsePadOpState {
    fn save_frame(&mut self, op: &PulsePad, input: &DeviceTensor, frame: usize) -> TractResult<()> {
        let mut frame_shape: TVec<usize> = input.shape().into();
        frame_shape[op.axis] = 1;
        let last_valid_frame = DeviceTensor::uninitialized_dt(input.datum_type(), &frame_shape)?;
        CUDA_STREAM.with(|stream| {
            device_tensor_assign_slice(
                stream,
                &last_valid_frame,
                0..1,
                input,
                frame..frame + 1,
                op.axis,
            )
        })?;
        self.last_valid_frame = Some(last_valid_frame);
        Ok(())
    }

    fn pad(
        &mut self,
        session: &SessionState,
        cuda_op: &CudaPulsePad,
        input: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let op = &cuda_op.op;
        let pulse = input.shape()[op.axis];
        let pulse_begin = self.current_pos;
        let pulse_end = self.current_pos + pulse;
        self.current_pos += pulse - op.overlap;
        let end_input =
            op.end_input.eval(&session.resolved_symbols).to_usize().unwrap_or(usize::MAX);
        let after = op.after.eval(&session.resolved_symbols).to_usize().unwrap_or(usize::MAX);

        if let PadMode::Edge = op.mode {
            if after != 0 && pulse_begin < end_input {
                let latest_valid_frame = (end_input - pulse_begin).min(pulse) - 1;
                Self::save_frame(self, op, input, latest_valid_frame)?;
            }
        }

        let mut output =
            make_tensor_for_node(session, self.node_id, input.datum_type(), input.shape())?;
        CUDA_STREAM.with(|stream| {
            stream.memcpy_dtod(&get_cuda_view(input), &mut get_cuda_view_mut(&output))
        })?;

        // pulse is entirely in either valid input or invalid input
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
                    &mut output,
                    cuda_op.device_cst.as_ref().unwrap(),
                    op.axis,
                    0..fill_up_to,
                )?,
                PadMode::Edge => fill_slice_repeating_one_frame(
                    &mut output,
                    input,
                    op.axis,
                    0..fill_up_to,
                    fill_up_to,
                )?,
                _ => unimplemented!(),
            }
        }
        if pulse_end > end_input && after > 0 {
            let fill_from = pulse - (pulse_end - end_input).min(pulse);
            match &op.mode {
                PadMode::Constant(_) => fill_slice_constant(
                    &mut output,
                    cuda_op.device_cst.as_ref().unwrap(),
                    op.axis,
                    fill_from..pulse,
                )?,

                PadMode::Edge => fill_slice_repeating_one_frame(
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

trivial_op_state_freeeze!(CudaPulsePadOpState);
