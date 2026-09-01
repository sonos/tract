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
    ) -> TractResult<(usize, usize, usize)> {
        ensure!(query.datum_type() == DatumType::F16, "GDN query must be F16");
        ensure!(key.datum_type() == DatumType::F16 && value.datum_type() == DatumType::F16);
        ensure!(beta.datum_type() == DatumType::F16, "GDN beta must be F16");
        ensure!(log_decay.datum_type() == DatumType::F32, "GDN decay must be F32");
        ensure!(initial_state.datum_type() == DatumType::F32, "GDN state must be F32");
        ensure!(query.shape() == key.shape() && query.shape() == value.shape());
        ensure!(query.rank() == 4, "GDN q/k/v layout is [b, S, h, w]");
        ensure!(query.shape()[0] == 1, "the cuda GDN step loop requires batch 1");
        let steps = query.shape()[1];
        let heads = query.shape()[2];
        let width = query.shape()[3];
        ensure!(width == 128, "the Qwen3.5 kernel currently requires width=128");
        ensure!(
            log_decay.len() == steps * heads && beta.len() == steps * heads,
            "GDN head mismatch: steps={steps}, heads={heads}, q={:?}, g={:?} (len {}), beta={:?} (len {})",
            query.shape(),
            log_decay.shape(),
            log_decay.len(),
            beta.shape(),
            beta.len(),
        );
        ensure!(initial_state.len() == heads * width * width);
        Ok((steps, heads, width))
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
        let (steps, heads, width) =
            self.validate(query, key, value, log_decay, beta, initial_state)?;
        ensure!(output.shape() == query.shape() && output.datum_type() == DatumType::F16);
        ensure!(
            final_state.shape() == initial_state.shape()
                && final_state.datum_type() == DatumType::F32
        );

        let function = cuda_context()
            .load_pipeline(LibraryName::GdnRecurrent, "tract_gdn_recurrent_f16".to_string())?;
        // The kernel is single-step; loop over S host-side with per-step
        // views (the [1, S, h, w] layout is contiguous per step) and
        // ping-pong the state so consecutive steps never read the buffer
        // they write. Buffer parity is chosen so the LAST step always
        // writes `final_state`, whatever the parity of `steps`.
        //
        // KNOWN LIMITATION: unlike Metal's chunked gated-delta-rule kernel,
        // this issues one launch per step, so a long CUDA prefill pays S
        // serialized kernel-launch overheads instead of ~S/GDN_CHUNK. A
        // chunked CUDA kernel is real follow-up work, not done here.
        let scratch_state = if steps > 1 {
            Some(unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, initial_state.shape())? })
        } else {
            None
        };
        let qkv_step = heads * width * DatumType::F16.size_of();
        let g_step = heads * DatumType::F32.size_of();
        let b_step = heads * DatumType::F16.size_of();
        let state_bytes = heads * width * width * DatumType::F32.size_of();
        for s in 0..steps {
            let in_state = if s == 0 {
                initial_state
            } else if (steps - s) % 2 == 1 {
                scratch_state.as_ref().unwrap()
            } else {
                final_state
            };
            let out_state =
                if (steps - s) % 2 == 1 { final_state } else { scratch_state.as_ref().unwrap() };
            let q = crate::kernels::get_sliced_cuda_view(query, s * qkv_step, qkv_step)?;
            let k = crate::kernels::get_sliced_cuda_view(key, s * qkv_step, qkv_step)?;
            let v = crate::kernels::get_sliced_cuda_view(value, s * qkv_step, qkv_step)?;
            let g = crate::kernels::get_sliced_cuda_view(log_decay, s * g_step, g_step)?;
            let b = crate::kernels::get_sliced_cuda_view(beta, s * b_step, b_step)?;
            let state = crate::kernels::get_sliced_cuda_view(in_state, 0, state_bytes)?;
            let out = crate::kernels::get_sliced_cuda_view(output, s * qkv_step, qkv_step)?;
            let next_state = crate::kernels::get_sliced_cuda_view(out_state, 0, state_bytes)?;
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
            })?;
        }
        Ok(())
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

#[allow(clippy::too_many_arguments)]
pub fn cuda_gdn_recurrent_launch(
    query: &DeviceTensor,
    key: &DeviceTensor,
    value: &DeviceTensor,
    log_decay: &DeviceTensor,
    beta: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
    sigmoid_beta: bool,
) -> TractResult<()> {
    anyhow::ensure!(!sigmoid_beta, "cuda GDN kernel has no in-kernel beta sigmoid");
    crate::with_cuda_stream(|stream| {
        CudaGdnRecurrent.dispatch_eval(
            stream,
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            output,
            final_state,
        )
    })
}

crate::register_cuda_op!(
    tract_transformers::ops::gdn_recurrent::GatedDeltaNetRecurrent,
    |source, node, op| {
        // No in-kernel beta sigmoid on cuda yet.
        if op.sigmoid_beta {
            return Ok(None);
        }
        // The dispatch loops the single-step kernel over S host-side, so
        // symbolic or concrete S are both fine. Still required: f16 dtypes,
        // ungrouped heads (q/k head count == v head count; no GQA-group
        // indexing in this kernel yet) and batch 1 ([b, S, h, w] layout).
        let facts = source.node_input_facts(node.id)?;
        let dts: Vec<DatumType> = facts.iter().map(|f| f.datum_type).collect();
        let rank_ok = facts[0].rank() == 4;
        let batch_ok = rank_ok && facts[0].shape[0] == 1.to_dim();
        let ungrouped = facts[0].shape == facts[2].shape;
        let dts_ok = dts
            == [
                DatumType::F16,
                DatumType::F16,
                DatumType::F16,
                DatumType::F32,
                DatumType::F16,
                DatumType::F32,
            ];
        if !rank_ok || !batch_ok || !ungrouped || !dts_ok {
            return Ok(None);
        }
        Ok(Some(Box::new(tract_gpu::ops::gdn_recurrent::GpuGatedDeltaNetRecurrent {
            backend_name: "Cuda",
            dispatch: cuda_gdn_recurrent_launch,
            sigmoid_beta: false,
        })))
    }
);

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
