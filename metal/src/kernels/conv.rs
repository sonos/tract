use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensor;

pub fn kernel_name(hw_rank: usize, dt: DatumType) -> TractResult<String> {
    let dt_name = if dt == DatumType::F16 { "f16" } else { "f32" };
    Ok(format!("conv{hw_rank}d_{dt_name}_generic"))
}

pub fn metal_conv_dispatch(
    stream: &MetalStream,
    op: &Conv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(input);
    stream.retain_tensor(weights);
    if let Some(b) = bias {
        stream.retain_tensor(b);
    }
    stream.retain_tensor(output);

    let input_shape = op.pool_spec.data_format.shape(input.shape())?;
    let hw_rank = input_shape.hw_rank();
    let func_name = kernel_name(hw_rank, input.datum_type())?;
    let pipeline = stream.load_pipeline(LibraryName::ConvOps, &func_name)?;

    let co_per_group = op.pool_spec.output_channels / op.group;
    let ci_per_group = op.pool_spec.input_channels / op.group;

    // in_shape: [N, C, spatial...]
    let in_n = *input_shape.n().unwrap_or(&1);
    let in_c = *input_shape.c();
    let mut in_shape_buf: TVec<i32> = tvec![in_n as i32, in_c as i32];
    in_shape_buf.extend(input_shape.hw_dims().iter().map(|&d| d as i32));

    let mut in_strides_buf: TVec<i32> =
        tvec![*input_shape.n_stride().unwrap_or(&0) as i32, *input_shape.c_stride() as i32];
    in_strides_buf.extend(input_shape.hw_strides().iter().map(|&s| s as i32));

    // ker_params: [groups, co_per_group, ci_per_group, ker_spatial...]
    let mut ker_params: TVec<i32> =
        tvec![op.group as i32, co_per_group as i32, ci_per_group as i32];
    ker_params.extend(weights.shape()[2..].iter().map(|&d| d as i32));

    // ker_strides: [g_stride, o_stride, i_stride, spatial...]
    let group_stride = weights.strides()[0] as usize * co_per_group;
    let mut ker_strides: TVec<i32> = tvec![group_stride as i32];
    ker_strides.extend(weights.strides().iter().map(|&s| s as i32));

    // padding
    let padding = op.pool_spec.computed_padding(input_shape.hw_dims());
    let pad_buf: TVec<i32> = padding.iter().map(|p| p.pad_before as i32).collect();

    let strides = op.pool_spec.strides();
    let strides_buf: TVec<i32> = strides.iter().map(|&s| s as i32).collect();

    let dilations = op.pool_spec.dilations();
    let dilations_buf: TVec<i32> = dilations.iter().map(|&d| d as i32).collect();

    let output_shape = op.pool_spec.data_format.shape(output.shape())?;
    let out_n = *output_shape.n().unwrap_or(&1);
    let out_c = *output_shape.c();
    let mut out_shape_buf: TVec<i32> = tvec![out_n as i32, out_c as i32];
    out_shape_buf.extend(output_shape.hw_dims().iter().map(|&d| d as i32));

    let mut out_strides_buf: TVec<i32> =
        tvec![*output_shape.n_stride().unwrap_or(&0) as i32, *output_shape.c_stride() as i32];
    out_strides_buf.extend(output_shape.hw_strides().iter().map(|&s| s as i32));

    // bias_stride: -1 means no bias, 0 means scalar broadcast, 1 means per-channel
    let bias_stride: i32 = if let Some(b) = bias { if b.rank() == 0 { 0 } else { 1 } } else { -1 };

    let spatial_out: usize = output_shape.hw_dims().iter().product();
    let threads_per_group = 32usize;

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_slice(1, &in_shape_buf);
        encoder.set_slice(2, &in_strides_buf);
        encoder.set_metal_tensor(3, weights, metal::MTLResourceUsage::Read);
        encoder.set_slice(4, &ker_params);
        encoder.set_slice(5, &ker_strides);
        if let Some(b) = bias {
            encoder.set_metal_tensor(6, b, metal::MTLResourceUsage::Read);
        } else {
            // Empty buffer — kernel checks bias_stride < 0
            encoder.set_bytes(6, 0, std::ptr::null());
        }
        encoder.set_slice(7, &[bias_stride]);
        encoder.set_slice(8, &pad_buf);
        encoder.set_slice(9, &strides_buf);
        encoder.set_slice(10, &dilations_buf);
        encoder.set_metal_tensor(11, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(12, &out_shape_buf);
        encoder.set_slice(13, &out_strides_buf);

        let grid_size = MTLSize {
            width: spatial_out.div_ceil(threads_per_group) as _,
            height: out_c as _,
            depth: out_n as _,
        };
        let group_size = MTLSize { width: threads_per_group as _, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}
