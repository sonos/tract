use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_core::ops::fft::Stft;
use tract_gpu::ops::stft::{FFT_LEN, GpuStft};
use tract_gpu::tensor::DeviceTensor;

/// Fused STFT on Metal: gather a strided 512-sample frame, apply the pre-padded window,
/// then a 512-point forward FFT (one FFT per threadgroup). `input` is interleaved-complex
/// f32 `[lead.., T, 2]`, `window` the real `[FFT_LEN]` window, `output`
/// `[lead.., frames, FFT_LEN, 2]`.
fn dispatch_eval(
    stream: &MetalStream,
    stride: usize,
    input: &DeviceTensor,
    window: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(input);
    stream.retain_tensor(window);
    stream.retain_tensor(output);

    ensure!(input.datum_type() == DatumType::F32 && output.datum_type() == DatumType::F32);
    ensure!(window.datum_type() == DatumType::F32 && window.len() == FFT_LEN);
    let rank = input.rank();
    ensure!(rank >= 2 && input.shape()[rank - 1] == 2, "STFT input must be [.., T, 2]");
    let axis = rank - 2;
    let batch: usize = input.shape()[..axis].iter().product();
    let t = input.shape()[axis];
    ensure!(
        output.rank() == rank + 1 && output.shape()[axis + 1] == FFT_LEN,
        "STFT output must be [.., frames, {FFT_LEN}, 2], got {:?}",
        output.shape()
    );
    let frames = output.shape()[axis];

    let pipeline = stream.load_pipeline(LibraryName::Fft, "stft512_forward")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, window, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
        encoder.set_bytes(3, 4, &(t as i32) as *const i32 as *const _);
        encoder.set_bytes(4, 4, &(frames as i32) as *const i32 as *const _);
        encoder.set_bytes(5, 4, &(stride as i32) as *const i32 as *const _);

        let grid_size =
            MTLSize { width: frames as NSUInteger, height: batch as NSUInteger, depth: 1 };
        let group_size = MTLSize { width: (FFT_LEN / 2) as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

/// Launch the Metal STFT kernel for the backend-agnostic [`GpuStft`].
pub fn metal_stft_dispatch(
    stride: usize,
    input: &DeviceTensor,
    window: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| dispatch_eval(stream, stride, input, window, output))
}

fn metal_stft_op(axis: usize, stride: usize, window: Arc<Tensor>) -> GpuStft {
    GpuStft { axis, stride, window, backend_name: "Metal", dispatch: metal_stft_dispatch }
}

crate::register_metal_op!(Stft, |source, node, op| {
    let in_fact = &source.node_input_facts(node.id)?[0];
    rule_if!(op.frame == FFT_LEN);
    rule_if!(in_fact.rank() >= 2 && op.axis == in_fact.rank() - 2);
    rule_if!(in_fact.datum_type == DatumType::F32);
    let window = tract_gpu::ops::stft::padded_window(op.window.as_ref())?;
    Ok(Some(Box::new(metal_stft_op(op.axis, op.stride, window))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::MetalTransform;
    use tract_core::transform::ModelTransform;

    #[test]
    fn stft_lowers_and_matches_cpu() -> TractResult<()> {
        let (t, frame, stride, win_len) = (2000usize, FFT_LEN, 160usize, 400usize);
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
        MetalTransform::default().transform(&mut gpu_model)?;
        assert!(
            gpu_model.nodes().iter().any(|n| n.op_as::<GpuStft>().is_some()),
            "transform did not lower Stft to GpuStft"
        );
        let gpu = gpu_model.into_runnable()?.run(tvec!(input.into_tvalue()))?;

        let cpu = cpu[0].to_plain_array_view::<f32>()?;
        let gpu = gpu[0].to_plain_array_view::<f32>()?;
        let (cpu, gpu) = (cpu.as_slice().unwrap(), gpu.as_slice().unwrap());
        assert_eq!(cpu.len(), gpu.len());
        let max_err = cpu.iter().zip(gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_err < 1e-2, "max err {max_err} between CPU and GPU STFT");
        Ok(())
    }
}
