use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

pub fn dispatch_routed_combine_f32(
    stream: &MetalStream,
    route_values: &DeviceTensor,
    route_token_ids: &DeviceTensor,
    route_weights: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(route_values);
    stream.retain_tensor(route_token_ids);
    stream.retain_tensor(route_weights);
    stream.retain_tensor(output);

    ensure!(route_values.rank() == 2, "route_values must be [routes, d_model]");
    ensure!(route_token_ids.rank() == 1, "route_token_ids must be [routes]");
    ensure!(route_weights.rank() == 1, "route_weights must be [routes]");
    ensure!(output.rank() == 2 || output.rank() == 3, "output must be rank 2 or 3");
    ensure!(route_values.datum_type() == f32::datum_type());
    ensure!(route_token_ids.datum_type() == i64::datum_type());
    ensure!(route_weights.datum_type() == f32::datum_type());
    ensure!(output.datum_type() == f32::datum_type());

    let route_count = route_token_ids.shape()[0];
    ensure!(route_values.shape()[0] == route_count);
    ensure!(route_weights.shape()[0] == route_count);

    let d_model = *output.shape().last().context("output has no feature axis")?;
    let token_count = output.len() / d_model;
    ensure!(
        route_values.shape()[1] == d_model,
        "route value dim {} does not match output dim {d_model}",
        route_values.shape()[1]
    );

    let route_count = route_count as u32;
    let token_count = token_count as u32;
    let d_model = d_model as u32;
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "routed_combine_f32")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, route_values, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, route_token_ids, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, route_weights, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[route_count]);
        encoder.set_slice(5, &[token_count]);
        encoder.set_slice(6, &[d_model]);

        let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}
