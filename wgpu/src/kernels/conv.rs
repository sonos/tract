use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    ChainStep, DW_WG, DepthwiseShape, EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey,
    ShaderDtype, conv_depthwise_module, conv_module, keys_for, pack_u32s, program_key,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Conv, &["conv2d"], shader_f16)
}

fn as_u32_i32(v: i32) -> u32 {
    v as u32
}

/// One filter per channel, no cross-channel sum: the tiled kernel applies.
fn depthwise_shape(op: &Conv, channels_last: bool) -> Option<DepthwiseShape> {
    let spec = &op.pool_spec;
    if channels_last
        || op.group == 1
        || op.group != spec.input_channels
        || spec.output_channels != spec.input_channels
        || spec.kernel_shape.len() != 2
    {
        return None;
    }
    let strides = spec.strides();
    let dilations = spec.dilations();
    Some(DepthwiseShape {
        kh: spec.kernel_shape[0] as u32,
        kw: spec.kernel_shape[1] as u32,
        stride_h: strides[0] as u32,
        stride_w: strides[1] as u32,
        dil_h: dilations[0] as u32,
        dil_w: dilations[1] as u32,
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_depthwise(
    q: &crate::WgpuQueue,
    op: &Conv,
    shape: DepthwiseShape,
    epilogue: &[ChainStep],
    input: &DeviceTensor,
    weights: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    let dt = ShaderDtype::from_datum(input.datum_type())?;
    let layout = LayoutKind::Chain(3 + extras.len() as u8);
    let key = format!("convdw{}", shape.key());
    let key = program_key(&key, dt, epilogue, extras.len());
    let pipeline =
        q.context().chain_pipeline(&key, layout, EntryPoint::typed("conv_dw", dt), || {
            conv_depthwise_module(dt, shape, epilogue, extras.len())
        })?;

    let in_shape = op.pool_spec.data_format.shape(input.shape())?;
    let out_shape = op.pool_spec.data_format.shape(output.shape())?;
    let n = *in_shape.n().unwrap_or(&1);
    let co = *out_shape.c();
    let (oh, ow) = (out_shape.hw_dims()[0], out_shape.hw_dims()[1]);
    let padding = op.pool_spec.computed_padding(in_shape.hw_dims());
    let ws = weights.strides();

    let mut buffers: Vec<&crate::context::WgpuBuffer> =
        vec![get_wgpu_buffer(input), get_wgpu_buffer(weights)];
    buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
    buffers.push(get_wgpu_buffer(output));
    let bg = q.context().bind_group(layout, &buffers, q.uniform())?;

    let mut off_extra = [0u32; 4];
    let mut mode_extra = [0u32; 4];
    for (i, t) in extras.iter().enumerate() {
        off_extra[i] = element_offset(t, 0) as u32;
        mode_extra[i] = crate::kernels::matmul::epilogue_mode(t, co)?;
    }
    let mut vals = vec![
        element_offset(input, 0) as u32,
        element_offset(weights, 0) as u32,
        element_offset(output, 0) as u32,
        n as u32,
        co as u32,
        in_shape.hw_dims()[0] as u32,
        in_shape.hw_dims()[1] as u32,
        oh as u32,
        ow as u32,
        padding[0].pad_before as u32,
        padding[1].pad_before as u32,
        0,
        as_u32_i32(*in_shape.n_stride().unwrap_or(&0) as i32),
        as_u32_i32(*in_shape.c_stride() as i32),
        as_u32_i32(in_shape.hw_strides()[0] as i32),
        as_u32_i32(in_shape.hw_strides()[1] as i32),
        as_u32_i32(ws[0] as i32),
        as_u32_i32(ws[2] as i32),
        as_u32_i32(ws[3] as i32),
        0,
        as_u32_i32(*out_shape.n_stride().unwrap_or(&0) as i32),
        as_u32_i32(*out_shape.c_stride() as i32),
        as_u32_i32(out_shape.hw_strides()[0] as i32),
        as_u32_i32(out_shape.hw_strides()[1] as i32),
    ];
    vals.extend_from_slice(&off_extra);
    vals.extend_from_slice(&mode_extra);
    let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
    let groups = [(ow as u32).div_ceil(DW_WG), (oh as u32).div_ceil(DW_WG), (n * co) as u32];
    q.dispatch_grid("conv_depthwise", &pipeline, &bg, dyn_off, groups)
}

pub fn wgpu_conv_dispatch(
    op: &Conv,
    epilogue: &[ChainStep],
    input: &DeviceTensor,
    weights: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(weights);
        for t in extras {
            q.retain_tensor(t);
        }
        q.retain_tensor(output);
        ensure!(matches!(input.datum_type(), DatumType::F32 | DatumType::F16));
        let channels_last = op.pool_spec.data_format.c_is_last();
        if let Some(shape) = depthwise_shape(op, channels_last) {
            return dispatch_depthwise(q, op, shape, epilogue, input, weights, extras, output);
        }
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(3 + extras.len() as u8);
        let pipeline = if epilogue.is_empty() {
            q.context().pipeline(PipelineKey {
                module: ModuleKey { kind: ModuleKind::Conv, dtype: dt },
                entry: EntryPoint::typed("conv2d", dt),
            })?
        } else {
            let key = program_key("conv", dt, epilogue, extras.len());
            q.context().chain_pipeline(&key, layout, EntryPoint::typed("conv2d", dt), || {
                conv_module(dt, epilogue, extras.len())
            })?
        };

        let in_shape = op.pool_spec.data_format.shape(input.shape())?;
        let out_shape = op.pool_spec.data_format.shape(output.shape())?;
        ensure!(in_shape.hw_rank() == 2, "tract-wgpu conv is 2D only");
        let n = *in_shape.n().unwrap_or(&1);
        let ci = *in_shape.c();
        let co = *out_shape.c();
        let ih = in_shape.hw_dims()[0];
        let iw = in_shape.hw_dims()[1];
        let oh = out_shape.hw_dims()[0];
        let ow = out_shape.hw_dims()[1];
        let kh = op.pool_spec.kernel_shape[0];
        let kw = op.pool_spec.kernel_shape[1];
        let groups = op.group;
        let ci_pg = op.pool_spec.input_channels / groups;
        let co_pg = op.pool_spec.output_channels / groups;
        let padding = op.pool_spec.computed_padding(in_shape.hw_dims());
        let strides = op.pool_spec.strides();
        let dilations = op.pool_spec.dilations();
        let channels_last = op.pool_spec.data_format.c_is_last() as u32;

        let in_sn = *in_shape.n_stride().unwrap_or(&0);
        let in_sc = *in_shape.c_stride();
        let in_sh = in_shape.hw_strides()[0];
        let in_sw = in_shape.hw_strides()[1];
        let out_sn = *out_shape.n_stride().unwrap_or(&0);
        let out_sc = *out_shape.c_stride();
        let out_sh = out_shape.hw_strides()[0];
        let out_sw = out_shape.hw_strides()[1];

        // OIHW: [co, ci_pg, kh, kw]
        let ws = weights.strides();
        ensure!(ws.len() >= 4, "expected 4D OIHW kernel, got {:?}", weights.shape());
        let w_so = ws[0];
        let w_si = ws[1];
        let w_sh = ws[2];
        let w_sw = ws[3];

        let mut buffers: Vec<&crate::context::WgpuBuffer> =
            vec![get_wgpu_buffer(input), get_wgpu_buffer(weights)];
        buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(weights, 0) as u32,
            element_offset(output, 0) as u32,
            n as u32,
            ci as u32,
            co as u32,
            ih as u32,
            iw as u32,
            oh as u32,
            ow as u32,
            kh as u32,
            kw as u32,
            groups as u32,
            ci_pg as u32,
            co_pg as u32,
            channels_last,
            padding[0].pad_before as u32,
            padding[1].pad_before as u32,
            strides[0] as u32,
            strides[1] as u32,
            dilations[0] as u32,
            dilations[1] as u32,
            as_u32_i32(in_sn as i32),
            as_u32_i32(in_sc as i32),
            as_u32_i32(in_sh as i32),
            as_u32_i32(in_sw as i32),
            as_u32_i32(w_so as i32),
            as_u32_i32(w_si as i32),
            as_u32_i32(w_sh as i32),
            as_u32_i32(w_sw as i32),
            as_u32_i32(out_sn as i32),
            as_u32_i32(out_sc as i32),
            as_u32_i32(out_sh as i32),
            as_u32_i32(out_sw as i32),
        ]);
        let mut tail = vec![0u32; 2];
        let mut off_extra = [0u32; 4];
        let mut mode_extra = [0u32; 4];
        for (i, t) in extras.iter().enumerate() {
            off_extra[i] = element_offset(t, 0) as u32;
            mode_extra[i] = crate::kernels::matmul::epilogue_mode(t, co)?;
        }
        tail.extend_from_slice(&off_extra);
        tail.extend_from_slice(&mode_extra);
        let mut params = params;
        params.extend_from_slice(&pack_u32s(&tail));
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("conv", &pipeline, &bg, dyn_off, (n * co * oh * ow) as u64)
    })
}
