use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, keys_for, pack_u32s,
    pad8_stride, pad8_u32,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Copy, &["copy"], shader_f16)
}

pub fn wgpu_copy_nd_dispatch(
    input: &DeviceTensor,
    input_offset: usize,
    input_strides: &[isize],
    output: &DeviceTensor,
    output_offset: usize,
    output_shape: &[usize],
    output_strides: &[isize],
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let key = PipelineKey {
            module: ModuleKey { kind: ModuleKind::Copy, dtype: dt },
            entry: EntryPoint::typed("copy", dt),
        };
        let pipeline = q.context().pipeline(key)?;
        let in_buf = get_wgpu_buffer(input);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(LayoutKind::Unary, &[in_buf, out_buf], q.uniform())?;
        let n: u64 = output_shape.iter().map(|d| *d as u64).product();
        let rank = output_shape.len().max(1);
        let mut vals = vec![
            element_offset(input, input_offset) as u32,
            element_offset(output, output_offset) as u32,
            rank as u32,
            n as u32,
        ];
        vals.extend_from_slice(&pad8_stride(input_strides));
        vals.extend_from_slice(&pad8_stride(output_strides));
        vals.extend_from_slice(&pad8_u32(output_shape));
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        q.dispatch("copy", &pipeline, &bg, dyn_off, n)
    })
}
