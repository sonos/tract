use tract_core::internal::*;
use tract_core::ops::cnn::Deconv;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, keys_for, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Deconv, &["conv_transpose2d"], shader_f16)
}

fn as_u32_i32(v: i32) -> u32 {
    v as u32
}

pub fn wgpu_deconv_dispatch(
    op: &Deconv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(weights);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::Deconv, dtype: dt },
            entry: EntryPoint::typed("conv_transpose2d", dt),
        })?;
        let in_shape = op.pool_spec.data_format.shape(input.shape())?;
        let out_shape = op.pool_spec.data_format.shape(output.shape())?;
        ensure!(in_shape.hw_rank() == 2, "tract-wgpu conv_transpose is 2D only");
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
        let padding = op.pool_spec.padding.compute_for_deconv(
            in_shape.hw_dims(),
            &op.pool_spec.kernel_shape,
            &op.pool_spec.dilations(),
            &op.pool_spec.strides(),
            &op.adjustments,
        )?;
        let strides = op.pool_spec.strides();
        let dilations = op.pool_spec.dilations();

        let in_sn = *in_shape.n_stride().unwrap_or(&0);
        let in_sc = *in_shape.c_stride();
        let in_sh = in_shape.hw_strides()[0];
        let in_sw = in_shape.hw_strides()[1];
        let out_sn = *out_shape.n_stride().unwrap_or(&0);
        let out_sc = *out_shape.c_stride();
        let out_sh = out_shape.hw_strides()[0];
        let out_sw = out_shape.hw_strides()[1];
        let ws = weights.strides();
        ensure!(ws.len() >= 4);
        // OIHW after tract's onnx converter: [co, ci_pg, kh, kw]
        let w_so = ws[0];
        let w_si = ws[1];
        let w_sh = ws[2];
        let w_sw = ws[3];

        let in_buf = get_wgpu_buffer(input);
        let w_buf = get_wgpu_buffer(weights);
        let out_buf = get_wgpu_buffer(output);
        let bg =
            q.context().bind_group(LayoutKind::Binary, &[in_buf, w_buf, out_buf], q.uniform())?;
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
            0,
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
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("deconv", &pipeline, &bg, dyn_off, (n * co * oh * ow) as u64)
    })
}
