use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};

#[derive(Debug, Clone)]
pub struct CudaCausalConv1dUpdate;

impl CudaCausalConv1dUpdate {
    pub fn validate(
        &self,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        state: &DeviceTensor,
    ) -> TractResult<(usize, usize)> {
        ensure!(input.datum_type() == DatumType::F16);
        ensure!(weight.datum_type() == DatumType::F16 && state.datum_type() == DatumType::F16);
        let channels = input.len();
        let kernel_width = *weight.shape().last().context("conv weight must have a kernel axis")?;
        ensure!(kernel_width == 4, "Qwen3.5 requires a four-tap convolution");
        ensure!(weight.len() == channels * kernel_width);
        ensure!(state.len() == channels * kernel_width);
        Ok((channels, kernel_width))
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        state: &DeviceTensor,
        output: &DeviceTensor,
        final_state: &DeviceTensor,
    ) -> TractResult<()> {
        let (channels, kernel_width) = self.validate(input, weight, state)?;
        ensure!(output.shape() == input.shape() && output.datum_type() == DatumType::F16);
        ensure!(final_state.shape() == state.shape() && final_state.datum_type() == DatumType::F16);
        let function = cuda_context().load_pipeline(
            LibraryName::GdnRecurrent,
            "tract_causal_conv1d_update_f16".to_string(),
        )?;
        let i = get_cuda_view(input);
        let w = get_cuda_view(weight);
        let s = get_cuda_view(state);
        let o = get_cuda_view(output);
        let ns = get_cuda_view(final_state);
        let mut args = TractLaunchArgs::new(stream, &function);
        args.push_view(&i);
        args.push_view(&w);
        args.push_view(&s);
        args.push_view(&o);
        args.push_view(&ns);
        args.push_i32(channels);
        args.push_i32(kernel_width);
        args.launch(LaunchConfig::for_num_elems(channels as u32))?;
        Ok(())
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        state: &DeviceTensor,
    ) -> TractResult<(DeviceTensor, DeviceTensor)> {
        let output = DeviceTensor::uninitialized_dt(DatumType::F16, input.shape())?;
        let final_state = DeviceTensor::uninitialized_dt(DatumType::F16, state.shape())?;
        self.dispatch_eval(stream, input, weight, state, &output, &final_state)?;
        stream.synchronize()?;
        Ok((output, final_state))
    }
}

pub fn cuda_causal_conv1d_update_launch(
    input: &DeviceTensor,
    weight: &DeviceTensor,
    state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        CudaCausalConv1dUpdate.dispatch_eval(stream, input, weight, state, output, final_state)
    })
}

crate::register_cuda_op!(
    tract_transformers::ops::causal_conv1d_update::CausalConv1dUpdate,
    |_source, _node, _op| {
        Ok(Some(Box::new(tract_gpu::ops::causal_conv1d_update::GpuCausalConv1dUpdate {
            backend_name: "Cuda",
            dispatch: cuda_causal_conv1d_update_launch,
        })))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn qwen35_conv_update_matches_cpu() -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
            let channels = 6144usize;
            let input_f = (0..channels).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect::<Vec<_>>();
            let weight_f =
                (0..channels * 4).map(|i| ((i % 13) as f32 - 6.0) / 64.0).collect::<Vec<_>>();
            let state_f =
                (0..channels * 4).map(|i| ((i % 11) as f32 - 5.0) / 32.0).collect::<Vec<_>>();
            let cvt = |v: &[f32]| v.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
            let input = Tensor::from_shape(&[1, channels], &cvt(&input_f))?.into_device()?;
            let weight = Tensor::from_shape(&[channels, 4], &cvt(&weight_f))?.into_device()?;
            let state = Tensor::from_shape(&[1, channels, 4], &cvt(&state_f))?.into_device()?;
            let (output, next) = CudaCausalConv1dUpdate.eval(stream, &input, &weight, &state)?;
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
}
