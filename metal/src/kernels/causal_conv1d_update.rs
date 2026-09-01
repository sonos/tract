use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

pub fn dispatch_eval(
    stream: &MetalStream,
    input: &DeviceTensor,
    weight: &DeviceTensor,
    state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    ensure!(input.datum_type() == DatumType::F16);
    ensure!(weight.datum_type() == DatumType::F16 && state.datum_type() == DatumType::F16);
    // Layout matches the CPU op: input [b, C, S], weight [C, k],
    // state [b, C, k].
    ensure!(input.rank() == 3, "conv input must be [b, C, S], got {:?}", input.shape());
    let (batch, channels, s_len) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let kernel_width = *weight.shape().last().context("conv weight must have a kernel axis")?;
    ensure!(kernel_width <= 8, "the Metal causal conv kernel supports at most 8 taps");
    ensure!(weight.len() == channels * kernel_width);
    ensure!(
        state.shape() == [batch, channels, kernel_width],
        "conv state must be [b, C, k], got {:?}",
        state.shape()
    );
    ensure!(output.shape() == input.shape() && output.datum_type() == DatumType::F16);
    ensure!(final_state.shape() == state.shape() && final_state.datum_type() == DatumType::F16);
    for tensor in [input, weight, state, output, final_state] {
        stream.retain_tensor(tensor);
    }
    let pipeline = stream.load_pipeline(LibraryName::GdnRecurrent, "causal_conv1d_update_f16")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, weight, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, state, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(4, final_state, metal::MTLResourceUsage::Write);
        let channels = channels as i32;
        let kernel_width = kernel_width as i32;
        let s_len = s_len as i32;
        let batch = batch as i32;
        encoder.set_bytes(5, size_of::<i32>() as u64, &channels as *const i32 as *const _);
        encoder.set_bytes(6, size_of::<i32>() as u64, &kernel_width as *const i32 as *const _);
        encoder.set_bytes(7, size_of::<i32>() as u64, &s_len as *const i32 as *const _);
        encoder.set_bytes(8, size_of::<i32>() as u64, &batch as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize { width: (batch * channels) as u64, height: 1, depth: 1 },
            MTLSize { width: 256.min(channels) as u64, height: 1, depth: 1 },
        );
    });
    Ok(())
}

pub fn metal_causal_conv1d_update_launch(
    input: &DeviceTensor,
    weight: &DeviceTensor,
    state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        dispatch_eval(stream, input, weight, state, output, final_state)
    })
}

crate::register_metal_op!(
    tract_transformers::ops::causal_conv1d_update::CausalConv1dUpdate,
    |source, node, _op| {
        // The Metal kernel is f16-only; other dtype mixes (e.g. an all-f32
        // test export) stay on the CPU op.
        let facts = source.node_input_facts(node.id)?;
        if facts.iter().any(|f| f.datum_type != DatumType::F16) {
            return Ok(None);
        }
        Ok(Some(Box::new(tract_gpu::ops::causal_conv1d_update::GpuCausalConv1dUpdate {
            backend_name: "Metal",
            dispatch: metal_causal_conv1d_update_launch,
        })))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::{DeviceTensorExt, IntoDevice};

    #[test]
    fn qwen35_conv_update_matches_cpu() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let channels = 257usize;
            let input_f = (0..channels).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect::<Vec<_>>();
            let weight_f =
                (0..channels * 4).map(|i| ((i % 13) as f32 - 6.0) / 64.0).collect::<Vec<_>>();
            let state_f =
                (0..channels * 4).map(|i| ((i % 11) as f32 - 5.0) / 32.0).collect::<Vec<_>>();
            let cvt = |v: &[f32]| v.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
            let input = Tensor::from_shape(&[1, channels, 1], &cvt(&input_f))?.into_device()?;
            let weight = Tensor::from_shape(&[channels, 4], &cvt(&weight_f))?.into_device()?;
            let state = Tensor::from_shape(&[1, channels, 4], &cvt(&state_f))?.into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, input.shape())?;
            let next = DeviceTensor::uninitialized_dt(DatumType::F16, state.shape())?;
            dispatch_eval(stream, &input, &weight, &state, &output, &next)?;
            stream.wait_until_completed()?;
            let output = output.to_host()?.into_tensor();
            let next = next.to_host()?.into_tensor();
            let got = unsafe { output.as_slice_unchecked::<f16>() };
            let got_state = unsafe { next.as_slice_unchecked::<f16>() };
            for c in 0..channels {
                let base = c * 4;
                assert_eq!(&got_state[base..base + 3], &cvt(&state_f[base + 1..base + 4]));
                assert_eq!(got_state[base + 3], f16::from_f32(input_f[c]));
                let sum = (0..3).map(|t| state_f[base + t + 1] * weight_f[base + t]).sum::<f32>()
                    + input_f[c] * weight_f[base + 3];
                assert_eq!(got[c], f16::from_f32(sum / (1.0 + (-sum).exp())));
            }
            Ok(())
        })
    }

    #[test]
    fn qwen35_conv_multi_step_matches_cpu_op() -> TractResult<()> {
        use tract_transformers::ops::causal_conv1d_update::CausalConv1dUpdate;
        with_borrowed_metal_stream(|stream| {
            let (b, channels, s_len, k) = (1usize, 33usize, 7usize, 4usize);
            let input_f = (0..b * channels * s_len)
                .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
                .collect::<Vec<_>>();
            let weight_f =
                (0..channels * k).map(|i| ((i % 13) as f32 - 6.0) / 64.0).collect::<Vec<_>>();
            let state_f =
                (0..b * channels * k).map(|i| ((i % 11) as f32 - 5.0) / 32.0).collect::<Vec<_>>();
            let cvt = |v: &[f32]| v.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
            let input = Tensor::from_shape(&[b, channels, s_len], &cvt(&input_f))?;
            let weight = Tensor::from_shape(&[channels, k], &cvt(&weight_f))?;
            let state = Tensor::from_shape(&[b, channels, k], &cvt(&state_f))?;

            let cpu = CausalConv1dUpdate.eval(
                &EvalContext::out_of_plan(),
                tvec![
                    input.clone().into_tvalue(),
                    weight.clone().into_tvalue(),
                    state.clone().into_tvalue(),
                ],
            )?;

            let input_d = input.into_device()?;
            let weight_d = weight.into_device()?;
            let state_d = state.into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, input_d.shape())?;
            let next = DeviceTensor::uninitialized_dt(DatumType::F16, state_d.shape())?;
            dispatch_eval(stream, &input_d, &weight_d, &state_d, &output, &next)?;
            stream.wait_until_completed()?;
            let output = output.to_host()?.into_tensor();
            let next = next.to_host()?.into_tensor();
            output
                .cast_to::<f32>()?
                .into_owned()
                .close_enough(&cpu[0].cast_to::<f32>()?.into_owned(), Approximation::Approximate)?;
            next.cast_to::<f32>()?
                .into_owned()
                .close_enough(&cpu[1].cast_to::<f32>()?.into_owned(), Approximation::Approximate)?;
            Ok(())
        })
    }
}
