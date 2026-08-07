use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[allow(clippy::too_many_arguments)]
fn dispatch_eval(
    stream: &MetalStream,
    query: &DeviceTensor,
    key: &DeviceTensor,
    value: &DeviceTensor,
    log_decay: &DeviceTensor,
    beta: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    ensure!(query.datum_type() == DatumType::F16);
    ensure!(key.datum_type() == DatumType::F16 && value.datum_type() == DatumType::F16);
    ensure!(log_decay.datum_type() == DatumType::F32);
    ensure!(beta.datum_type() == DatumType::F16);
    ensure!(matches!(initial_state.datum_type(), DatumType::F16 | DatumType::F32));
    ensure!(query.shape() == key.shape() && query.shape() == value.shape());
    // Layout matches the CPU op: q/k/v [b, S, h, w], gates [b, S, h],
    // state [b, h, w, w].
    ensure!(query.rank() == 4, "GDN query must be [b, S, h, w], got {:?}", query.shape());
    let (batch, s_len, heads, width) =
        (query.shape()[0], query.shape()[1], query.shape()[2], query.shape()[3]);
    ensure!(log_decay.len() == batch * s_len * heads && beta.len() == batch * s_len * heads);
    ensure!(
        initial_state.shape() == [batch, heads, width, width],
        "GDN state must be [b, h, w, w], got {:?}",
        initial_state.shape()
    );
    ensure!(output.shape() == query.shape() && output.datum_type() == DatumType::F16);
    ensure!(final_state.shape() == initial_state.shape());

    for tensor in [query, key, value, log_decay, beta, initial_state, output, final_state] {
        stream.retain_tensor(tensor);
    }
    let kernel_name = if initial_state.datum_type() == DatumType::F16 {
        "gdn_recurrent_f16_state_f16"
    } else {
        "gdn_recurrent_f16"
    };
    let pipeline = stream.load_pipeline(LibraryName::GdnRecurrent, kernel_name)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        for (ix, tensor) in [query, key, value, log_decay, beta, initial_state].iter().enumerate() {
            encoder.set_metal_tensor(ix as u64, tensor, metal::MTLResourceUsage::Read);
        }
        encoder.set_metal_tensor(6, output, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(7, final_state, metal::MTLResourceUsage::Write);
        let heads = heads as i32;
        let width = width as i32;
        let s_len = s_len as i32;
        let batch = batch as i32;
        encoder.set_bytes(8, size_of::<i32>() as u64, &heads as *const i32 as *const _);
        encoder.set_bytes(9, size_of::<i32>() as u64, &width as *const i32 as *const _);
        encoder.set_bytes(10, size_of::<i32>() as u64, &s_len as *const i32 as *const _);
        encoder.set_bytes(11, size_of::<i32>() as u64, &batch as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize { width: (batch * heads * width) as u64, height: 1, depth: 1 },
            MTLSize { width: width.min(1024) as u64, height: 1, depth: 1 },
        );
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn metal_gdn_recurrent_launch(
    query: &DeviceTensor,
    key: &DeviceTensor,
    value: &DeviceTensor,
    log_decay: &DeviceTensor,
    beta: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        dispatch_eval(
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

crate::register_metal_op!(
    tract_transformers::ops::gdn_recurrent::GatedDeltaNetRecurrent,
    |source, node, _op| {
        // The Metal kernel is f16-only (q/k/v/beta f16, gates/state f32);
        // other dtype mixes (e.g. an all-f32 test export) stay on the CPU op.
        let facts = source.node_input_facts(node.id)?;
        let dts: Vec<DatumType> = facts.iter().map(|f| f.datum_type).collect();
        // q/k/v/beta f16, log_decay f32; the recurrent state may be f16
        // (graph exported with -idt f16) or f32, each has its kernel.
        if dts[..5] != [
            DatumType::F16,
            DatumType::F16,
            DatumType::F16,
            DatumType::F32,
            DatumType::F16,
        ] || !matches!(dts[5], DatumType::F16 | DatumType::F32)
        {
            return Ok(None);
        }
        Ok(Some(Box::new(tract_gpu::ops::gdn_recurrent::GpuGatedDeltaNetRecurrent {
            backend_name: "Metal",
            dispatch: metal_gdn_recurrent_launch,
        })))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::{DeviceTensorExt, IntoDevice};

    #[test]
    fn qwen35_recurrent_step_matches_cpu() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
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
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, q.shape())?;
            let next = DeviceTensor::uninitialized_dt(DatumType::F32, state.shape())?;
            dispatch_eval(stream, &q, &k, &v, &g, &beta, &state, &output, &next)?;
            stream.wait_until_completed()?;
            let output = output.to_host()?.into_tensor();
            let next = next.to_host()?.into_tensor();
            let got_output = unsafe { output.as_slice_unchecked::<f16>() };
            let got_state = unsafe { next.as_slice_unchecked::<f32>() };
            for head in 0..heads {
                let base = head * width;
                let matrix = head * width * width;
                let qi =
                    1.0 / (qf[base..base + width].iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                let ki =
                    1.0 / (kf[base..base + width].iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
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
                            (got_state[matrix + row * width + col] - expected_state).abs() < 3e-5
                        );
                        expected_output += qf[base + row] * qi * expected_state;
                    }
                    assert_eq!(
                        got_output[base + col],
                        f16::from_f32(expected_output / (width as f32).sqrt())
                    );
                }
            }
            Ok(())
        })
    }

    #[test]
    fn qwen35_recurrent_multi_step_matches_cpu_op() -> TractResult<()> {
        use tract_transformers::ops::gdn_recurrent::GatedDeltaNetRecurrent;
        with_borrowed_metal_stream(|stream| {
            let (b, s_len, heads, width) = (1usize, 5usize, 2usize, 32usize);
            let n_vec = b * s_len * heads * width;
            let n_gate = b * s_len * heads;
            let n_state = b * heads * width * width;
            let qf = (0..n_vec).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect::<Vec<_>>();
            let kf = (0..n_vec).map(|i| ((i % 29) as f32 - 14.0) / 64.0).collect::<Vec<_>>();
            let vf = (0..n_vec).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect::<Vec<_>>();
            let sf = (0..n_state).map(|i| ((i % 37) as f32 - 18.0) / 256.0).collect::<Vec<_>>();
            let gf = (0..n_gate).map(|i| -0.05 - 0.1 * (i % 7) as f32).collect::<Vec<_>>();
            let bf = (0..n_gate).map(|i| 0.125 + 0.08 * (i % 9) as f32).collect::<Vec<_>>();
            let as_f16 = |v: &[f32]| v.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
            let q = Tensor::from_shape(&[b, s_len, heads, width], &as_f16(&qf))?;
            let k = Tensor::from_shape(&[b, s_len, heads, width], &as_f16(&kf))?;
            let v = Tensor::from_shape(&[b, s_len, heads, width], &as_f16(&vf))?;
            let g = Tensor::from_shape(&[b, s_len, heads], &gf)?;
            let beta = Tensor::from_shape(&[b, s_len, heads], &as_f16(&bf))?;
            let state = Tensor::from_shape(&[b, heads, width, width], &sf)?;

            let cpu = GatedDeltaNetRecurrent.eval(tvec![
                q.clone().into_tvalue(),
                k.clone().into_tvalue(),
                v.clone().into_tvalue(),
                g.clone().into_tvalue(),
                beta.clone().into_tvalue(),
                state.clone().into_tvalue(),
            ])?;

            let qd = q.into_device()?;
            let kd = k.into_device()?;
            let vd = v.into_device()?;
            let gd = g.into_device()?;
            let betad = beta.into_device()?;
            let stated = state.into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, qd.shape())?;
            let next = DeviceTensor::uninitialized_dt(DatumType::F32, stated.shape())?;
            dispatch_eval(stream, &qd, &kd, &vd, &gd, &betad, &stated, &output, &next)?;
            stream.wait_until_completed()?;

            let output = output.to_host()?.into_tensor();
            let next = next.to_host()?.into_tensor();
            output
                .cast_to::<f32>()?
                .close_enough(&cpu[0].cast_to::<f32>()?.into_owned(), Approximation::Approximate)?;
            next.close_enough(&cpu[1].clone().into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }
}
