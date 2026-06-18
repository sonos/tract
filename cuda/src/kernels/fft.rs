use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::ops::fft::Stft;
use tract_gpu::ops::stft::{GpuStft, is_supported_frame};
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};

/// Forward complex FFT, one transform per CUDA block, over the innermost-but-one axis.
/// Input and output are interleaved-complex f32 `[batch.., N, 2]` (N a supported power of
/// two) with the trailing `N x 2` block contiguous. The kernel is picked by N.
#[derive(Debug, Clone)]
pub struct CudaFft;

impl CudaFft {
    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(input.datum_type() == DatumType::F32 && output.datum_type() == DatumType::F32);
        ensure!(input.shape() == output.shape());
        let rank = input.rank();
        ensure!(rank >= 2 && input.shape()[rank - 1] == 2, "CudaFft expects [batch.., N, 2]");
        let n = input.shape()[rank - 2];
        ensure!(is_supported_frame(n), "CudaFft: unsupported FFT length {n}");
        let batch = input.len() / (n * 2);

        let func = cuda_context().load_pipeline(LibraryName::Fft, format!("fft{n}_forward"))?;
        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let mut args = TractLaunchArgs::new(stream, &func);
        args.push_view(&i_view);
        args.push_view(&o_view);

        let cfg = LaunchConfig {
            grid_dim: (batch as _, 1, 1),
            block_dim: (n as u32 / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        args.launch(cfg)
    }
}

/// Fused STFT: frame + window + forward FFT, frame a supported power of two. `input` is
/// interleaved-complex f32 with the time axis at `rank-2` and the complex pair last
/// (`[lead.., T, 2]`); `window` is the pre-padded real window `[frame]`; `output` is
/// `[lead.., frames, frame, 2]`. The frame length is read from the output shape.
#[derive(Debug, Clone)]
pub struct CudaStft {
    pub stride: usize,
}

impl CudaStft {
    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        window: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(input.datum_type() == DatumType::F32 && output.datum_type() == DatumType::F32);
        ensure!(window.datum_type() == DatumType::F32);
        let rank = input.rank();
        ensure!(rank >= 2 && input.shape()[rank - 1] == 2, "STFT input must be [.., T, 2]");
        let axis = rank - 2;
        ensure!(output.rank() == rank + 1, "STFT output must be [.., frames, frame, 2]");
        let n = output.shape()[axis + 1];
        ensure!(is_supported_frame(n), "CudaStft: unsupported frame {n}");
        ensure!(window.len() == n, "STFT window len {} != frame {n}", window.len());
        let batch: usize = input.shape()[..axis].iter().product();
        let t = input.shape()[axis];
        let frames = output.shape()[axis];

        let func = cuda_context().load_pipeline(LibraryName::Fft, format!("stft{n}_forward"))?;
        let i_view = get_cuda_view(input);
        let w_view = get_cuda_view(window);
        let o_view = get_cuda_view(output);

        let mut args = TractLaunchArgs::new(stream, &func);
        args.push_view(&i_view);
        args.push_view(&w_view);
        args.push_view(&o_view);
        args.push_i32(t);
        args.push_i32(frames);
        args.push_i32(self.stride);

        let cfg = LaunchConfig {
            grid_dim: (frames as _, batch as _, 1),
            block_dim: (n as u32 / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        args.launch(cfg)
    }
}

/// Launch the CUDA STFT kernel for the backend-agnostic [`GpuStft`].
pub fn cuda_stft_dispatch(
    stride: usize,
    input: &DeviceTensor,
    window: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        CudaStft { stride }.dispatch_eval(stream, input, window, output)
    })
}

fn cuda_stft_op(axis: usize, frame: usize, stride: usize, window: Arc<Tensor>) -> GpuStft {
    GpuStft { axis, frame, stride, window, backend_name: "Cuda", dispatch: cuda_stft_dispatch }
}

crate::register_cuda_op!(Stft, |source, node, op| {
    let in_fact = &source.node_input_facts(node.id)?[0];
    rule_if!(is_supported_frame(op.frame));
    rule_if!(in_fact.rank() >= 2 && op.axis == in_fact.rank() - 2);
    rule_if!(in_fact.datum_type == DatumType::F32);
    let window = tract_gpu::ops::stft::padded_window(op.window.as_ref(), op.frame)?;
    Ok(Some(Box::new(cuda_stft_op(op.axis, op.frame, op.stride, window))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_cuda_stream;
    use num_complex::Complex;
    use rustfft::FftPlanner;
    use tract_gpu::ops::stft::SUPPORTED_FRAMES;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn fft_matches_rustfft() -> TractResult<()> {
        for &n in &SUPPORTED_FRAMES {
            let batch = 4;
            let mut data = vec![0f32; batch * n * 2];
            let mut frames: Vec<Vec<Complex<f32>>> = Vec::new();
            for b in 0..batch {
                let mut fr = Vec::with_capacity(n);
                for i in 0..n {
                    let re = (0.1 * i as f32 + 0.5 * b as f32).sin();
                    let im = (0.02 * i as f32).cos();
                    data[(b * n + i) * 2] = re;
                    data[(b * n + i) * 2 + 1] = im;
                    fr.push(Complex::new(re, im));
                }
                frames.push(fr);
            }

            let got = with_cuda_stream(|stream| {
                let input = Tensor::from_shape(&[batch, n, 2], &data)?.into_device()?;
                let output =
                    unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[batch, n, 2])? };
                CudaFft.dispatch_eval(stream, &input, &output)?;
                stream.synchronize()?;
                Ok(output.to_host()?.into_tensor())
            })?;
            let got = got.try_as_plain()?;
            let got = got.as_slice::<f32>()?;

            let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
            for b in 0..batch {
                let mut fr = frames[b].clone();
                fft.process(&mut fr);
                for k in 0..n {
                    let (gr, gi) = (got[(b * n + k) * 2], got[(b * n + k) * 2 + 1]);
                    let tol = 2e-2 * (n as f32).sqrt();
                    assert!(
                        (gr - fr[k].re).abs() < tol && (gi - fr[k].im).abs() < tol,
                        "n{n} b{b} k{k}: gpu=({gr},{gi}) ref=({},{})",
                        fr[k].re,
                        fr[k].im
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn stft_matches_core_stft() -> TractResult<()> {
        for &frame in &SUPPORTED_FRAMES {
            let (b, stride, win_len) = (2usize, 160usize, frame * 3 / 4);
            let t = frame + stride * 8;
            let frames = (t - frame) / stride + 1;

            let mut sig = vec![0f32; b * t * 2];
            for bi in 0..b {
                for i in 0..t {
                    sig[(bi * t + i) * 2] = (0.05 * i as f32 + 0.3 * bi as f32).sin();
                }
            }
            let signal = Tensor::from_shape(&[b, t, 2], &sig)?;

            let win: Vec<f32> = (0..win_len).map(|k| 0.5 - 0.5 * (k as f32).cos() * 0.01).collect();
            let pad_left = (frame - win_len) / 2;
            let mut winp = vec![0f32; frame];
            winp[pad_left..pad_left + win_len].copy_from_slice(&win);

            let stft = Stft {
                axis: 1,
                frame,
                stride,
                window: Some(std::sync::Arc::new(Tensor::from_shape(&[win_len], &win)?)),
            };
            let reference = stft.eval(tvec!(signal.clone().into_tvalue()))?;
            let reference = reference[0].to_plain_array_view::<f32>()?;
            let reference = reference.as_slice().unwrap();

            let got = with_cuda_stream(|stream| {
                let input = signal.clone().into_device()?;
                let window = Tensor::from_shape(&[frame], &winp)?.into_device()?;
                let output = unsafe {
                    DeviceTensor::uninitialized_dt(DatumType::F32, &[b, frames, frame, 2])?
                };
                CudaStft { stride }.dispatch_eval(stream, &input, &window, &output)?;
                stream.synchronize()?;
                Ok(output.to_host()?.into_tensor())
            })?;
            let got = got.try_as_plain()?;
            let got = got.as_slice::<f32>()?;

            let mut max_err = 0f32;
            for (g, r) in got.iter().zip(reference.iter()) {
                max_err = max_err.max((g - r).abs());
            }
            assert!(max_err < 5e-2, "frame {frame}: max err {max_err} vs core Stft");
        }
        Ok(())
    }

    #[test]
    fn stft_lowers_and_matches_cpu() -> TractResult<()> {
        use crate::transform::CudaTransform;
        use tract_core::transform::ModelTransform;

        for &frame in &[256usize, 1024] {
            let (stride, win_len) = (160usize, 400usize.min(frame));
            let t = frame + stride * 6;
            let win: Vec<f32> = (0..win_len)
                .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / win_len as f32).cos())
                .collect();

            let build = || -> TractResult<TypedModel> {
                let mut m = TypedModel::default();
                let src = m.add_source("sig", f32::fact([1, t, 2]))?;
                let stft = Stft { axis: 1, frame, stride, window: Some(Arc::new(tensor1(&win))) };
                let out = m.wire_node("stft", stft, &[src])?;
                m.select_output_outlets(&out)?;
                Ok(m)
            };

            let mut sig = vec![0f32; t * 2];
            for i in 0..t {
                sig[i * 2] = (0.05 * i as f32).sin();
            }
            let input = Tensor::from_shape(&[1, t, 2], &sig)?;

            let cpu = build()?.into_runnable()?.run(tvec!(input.clone().into_tvalue()))?;

            let mut gpu_model = build()?;
            CudaTransform.transform(&mut gpu_model)?;
            assert!(
                gpu_model.nodes().iter().any(|n| n.op_as::<GpuStft>().is_some()),
                "frame {frame}: transform did not lower Stft to GpuStft"
            );
            let gpu = gpu_model.into_runnable()?.run(tvec!(input.into_tvalue()))?;

            let cpu = cpu[0].to_plain_array_view::<f32>()?;
            let gpu = gpu[0].to_plain_array_view::<f32>()?;
            let (cpu, gpu) = (cpu.as_slice().unwrap(), gpu.as_slice().unwrap());
            assert_eq!(cpu.len(), gpu.len());
            let max_err = cpu.iter().zip(gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(max_err < 5e-2, "frame {frame}: max err {max_err} CPU vs GPU");
        }
        Ok(())
    }
}
