use derive_new::new;
use num_traits::One;
use tract_core::{internal::*, trivial_op_state_freeeze};
use tract_gpu::session_handler::make_tensor_for_node;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt};
use tract_pulse_opl::ops::Delay;

use crate::kernels::{
    get_cuda_view, get_cuda_view_mut, get_sliced_cuda_view, get_sliced_cuda_view_mut,
};
use crate::tensor::device_tensor_assign_slice;
use crate::CUDA_STREAM;

#[derive(Debug, new, Clone)]
pub struct CudaDelay(Delay);

impl Op for CudaDelay {
    fn name(&self) -> StaticName {
        "CudaDelay".into()
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
                assert!(input.shape()[0..op.axis].iter().all(|d| d.is_one()));
                let dt = input.datum_type();
                let frame_len = buffer.shape().iter().skip(op.axis + 1).product::<usize>();
                let offset = frame_len * input_pulse * dt.size_of();
                let to_shift = (buffered - input_pulse) * frame_len * dt.size_of();
                let scratch = DeviceTensor::uninitialized_dt(u8::datum_type(), &[to_shift])?;
                stream.memcpy_dtod(
                    &get_sliced_cuda_view(buffer, offset, to_shift)?,
                    &mut get_cuda_view_mut(&scratch),
                )?;
                stream.memcpy_dtod(
                    &get_cuda_view(&scratch),
                    &mut get_sliced_cuda_view_mut(buffer, 0, to_shift)?,
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
        let input_pulse = input.shape()[op.axis];
        let output_pulse = input_pulse + op.overlap;
        let mut output_shape: TVec<usize> = input.shape().into();
        output_shape[op.axis] = output_pulse;
        let device_input = input.as_device_tensor().context("Expected a cuda tensor")?;
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
            self.apply_delay_unchecked(op, &device_input, &mut output)?;
            Ok(tvec!(output.into_opaque_tensor().into()))
        }
    }
}

trivial_op_state_freeeze!(CudaDelayState);
