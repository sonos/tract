use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};

#[derive(Debug, Clone)]
pub struct CudaGdnRecurrent;

impl CudaGdnRecurrent {
    pub fn validate(
        &self,
        query: &DeviceTensor,
        key: &DeviceTensor,
        value: &DeviceTensor,
        log_decay: &DeviceTensor,
        beta: &DeviceTensor,
        initial_state: &DeviceTensor,
    ) -> TractResult<(usize, usize)> {
        ensure!(query.datum_type() == DatumType::F16, "GDN query must be F16");
        ensure!(key.datum_type() == DatumType::F16 && value.datum_type() == DatumType::F16);
        ensure!(beta.datum_type() == DatumType::F16, "GDN beta must be F16");
        ensure!(log_decay.datum_type() == DatumType::F32, "GDN decay must be F32");
        ensure!(initial_state.datum_type() == DatumType::F32, "GDN state must be F32");
        ensure!(query.shape() == key.shape() && query.shape() == value.shape());
        let width = *query.shape().last().context("GDN query must have a last axis")?;
        ensure!(width == 128, "the Qwen3.5 kernel currently requires width=128");
        ensure!(query.len().is_multiple_of(width));
        let heads = query.len() / width;
        ensure!(
            log_decay.len() == heads && beta.len() == heads,
            "GDN head mismatch: heads={heads}, q={:?}, g={:?} (len {}), beta={:?} (len {})",
            query.shape(),
            log_decay.shape(),
            log_decay.len(),
            beta.shape(),
            beta.len(),
        );
        ensure!(initial_state.len() == heads * width * width);
        Ok((heads, width))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        query: &DeviceTensor,
        key: &DeviceTensor,
        value: &DeviceTensor,
        log_decay: &DeviceTensor,
        beta: &DeviceTensor,
        initial_state: &DeviceTensor,
        output: &DeviceTensor,
        final_state: &DeviceTensor,
    ) -> TractResult<()> {
        let (heads, width) = self.validate(query, key, value, log_decay, beta, initial_state)?;
        ensure!(output.shape() == query.shape() && output.datum_type() == DatumType::F16);
        ensure!(
            final_state.shape() == initial_state.shape()
                && final_state.datum_type() == DatumType::F32
        );

        let function = cuda_context()
            .load_pipeline(LibraryName::GdnRecurrent, "tract_gdn_recurrent_f16".to_string())?;
        let q = get_cuda_view(query);
        let k = get_cuda_view(key);
        let v = get_cuda_view(value);
        let g = get_cuda_view(log_decay);
        let b = get_cuda_view(beta);
        let state = get_cuda_view(initial_state);
        let out = get_cuda_view(output);
        let next_state = get_cuda_view(final_state);
        let mut args = TractLaunchArgs::new(stream, &function);
        args.push_view(&q);
        args.push_view(&k);
        args.push_view(&v);
        args.push_view(&g);
        args.push_view(&b);
        args.push_view(&state);
        args.push_view(&out);
        args.push_view(&next_state);
        args.push_i32(heads);
        args.push_i32(width);
        args.launch(LaunchConfig {
            grid_dim: (heads as u32, 1, 1),
            block_dim: (width as u32, 1, 1),
            shared_mem_bytes: (3 * width * size_of::<f32>()) as u32,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval(
        &self,
        stream: &TractCudaStream,
        query: &DeviceTensor,
        key: &DeviceTensor,
        value: &DeviceTensor,
        log_decay: &DeviceTensor,
        beta: &DeviceTensor,
        initial_state: &DeviceTensor,
    ) -> TractResult<(DeviceTensor, DeviceTensor)> {
        self.validate(query, key, value, log_decay, beta, initial_state)?;
        let output = unsafe { DeviceTensor::uninitialized_dt(DatumType::F16, query.shape())? };
        let final_state =
            unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, initial_state.shape())? };
        self.dispatch_eval(
            stream,
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            &output,
            &final_state,
        )?;
        stream.synchronize()?;
        Ok((output, final_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn qwen35_recurrent_step_matches_cpu() -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
            let heads = 2usize;
            let width = 128usize;
            let vector_len = heads * width;
            let state_len = heads * width * width;
            let qf = (0..vector_len).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect::<Vec<_>>();
            let kf = (0..vector_len).map(|i| ((i % 29) as f32 - 14.0) / 64.0).collect::<Vec<_>>();
            let vf = (0..vector_len).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect::<Vec<_>>();
            let sf = (0..state_len).map(|i| ((i % 37) as f32 - 18.0) / 256.0).collect::<Vec<_>>();
            let gf = vec![-0.125f32, -0.75];
            let bf = vec![0.25f32, 0.875];
            let as_f16 = |v: &[f32]| v.iter().copied().map(f16::from_f32).collect::<Vec<_>>();

            let q = Tensor::from_shape(&[1, 1, heads, width], &as_f16(&qf))?.into_device()?;
            let k = Tensor::from_shape(&[1, 1, heads, width], &as_f16(&kf))?.into_device()?;
            let v = Tensor::from_shape(&[1, 1, heads, width], &as_f16(&vf))?.into_device()?;
            let g = Tensor::from_shape(&[1, 1, heads], &gf)?.into_device()?;
            let beta = Tensor::from_shape(&[1, 1, heads], &as_f16(&bf))?.into_device()?;
            let state = Tensor::from_shape(&[1, heads, width, width], &sf)?.into_device()?;
            let (output, next_state) =
                CudaGdnRecurrent.eval(stream, &q, &k, &v, &g, &beta, &state)?;
            let output = output.to_host()?.into_tensor();
            let next_state = next_state.to_host()?.into_tensor();
            let got_output = unsafe { output.as_slice_unchecked::<f16>() };
            let got_state = unsafe { next_state.as_slice_unchecked::<f32>() };

            for head in 0..heads {
                let base = head * width;
                let matrix = head * width * width;
                let q_norm = qf[base..base + width].iter().map(|x| x * x).sum::<f32>();
                let k_norm = kf[base..base + width].iter().map(|x| x * x).sum::<f32>();
                let qi = 1.0 / (q_norm + 1e-6).sqrt();
                let ki = 1.0 / (k_norm + 1e-6).sqrt();
                let decay = gf[head].exp();
                for col in 0..width {
                    let predicted = (0..width)
                        .map(|row| kf[base + row] * ki * sf[matrix + row * width + col] * decay)
                        .sum::<f32>();
                    let residual = (vf[base + col] - predicted) * bf[head];
                    let mut expected_output = 0.0f32;
                    for row in 0..width {
                        let expected_state =
                            sf[matrix + row * width + col] * decay + kf[base + row] * ki * residual;
                        assert!(
                            (got_state[matrix + row * width + col] - expected_state).abs() < 2e-6
                        );
                        expected_output += qf[base + row] * qi * expected_state;
                    }
                    let expected_output = f16::from_f32(expected_output / (width as f32).sqrt());
                    assert_eq!(got_output[base + col], expected_output);
                }
            }
            Ok(())
        })
    }
}
