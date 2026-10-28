use crate::LibraryName;
use crate::encoder::EncoderExt;
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

const MAX_THREADS: usize = 256;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_names() -> Vec<String> {
    ["f32", "f16"].into_iter().map(|t| format!("nn_ops::layer_norm_{t}")).collect()
}

pub fn metal_layer_norm_dispatch(
    input: &DeviceTensor,
    affine: Option<(&DeviceTensor, &DeviceTensor)>,
    axis: usize,
    eps: f32,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(is_supported_dt(input.datum_type()));
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    if let Some((scale, bias)) = affine {
        for t in [scale, bias] {
            ensure!(t.datum_type() == DatumType::F32 && t.len() == input.shape()[axis]);
        }
    }
    crate::with_metal_stream(|stream| {
        stream.retain_tensor(input);
        stream.retain_tensor(output);
        let (scale, bias) = affine.unwrap_or((output, output));
        stream.retain_tensor(scale);
        stream.retain_tensor(bias);
        let tname = DeviceTensor::tname(input.datum_type())?;
        let pipeline =
            stream.load_pipeline(LibraryName::NNOps, &format!("nn_ops::layer_norm_{tname}"))?;
        let shape = tract_gpu::utils::reshape_to_rank_3(input.shape(), axis);
        let dims = [shape[0], shape[1], shape[2], affine.is_some() as usize].map(|d| d as i32);
        let threads = shape[1].next_multiple_of(32).min(MAX_THREADS);
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, scale, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(2, bias, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(4, &dims);
            encoder.set_slice(5, &[eps]);
            encoder.set_threadgroup_memory_length(0, 32 * 4);
            encoder.dispatch_thread_groups(
                MTLSize { width: (shape[0] * shape[2]) as _, height: 1, depth: 1 },
                MTLSize { width: threads as _, height: 1, depth: 1 },
            );
        });
        Ok(())
    })
}

crate::register_metal_op!(tract_core::ops::nn::LayerNorm, |source, node, op| {
    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(is_supported_dt(input_facts[0].datum_type));
    rule_if!(input_facts[1..].iter().all(|f| f.datum_type == DatumType::F32));
    Ok(Some(Box::new(tract_gpu::ops::layer_norm::GpuLayerNorm {
        axis: op.axis,
        eps: op.eps.clone(),
        backend_name: "Metal",
        dispatch: metal_layer_norm_dispatch,
    })))
});
