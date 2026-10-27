use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::TransposeLaunch;
use tract_gpu::utils::merge_copy_axes;
use tract_gpu::utils::transpose_launch;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, keys_for,
    pack_u32s, pad8_stride, pad8_u32,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

/// Axes the copy shader's `Params` carries a shape and two strides for.
const PARAM_AXES: usize = 8;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    let mut keys = keys_for(ModuleKind::Copy, &["copy"], shader_f16);
    keys.extend(keys_for(ModuleKind::Transpose, &["copy_transpose"], shader_f16));
    keys
}

/// Copies `input` into `output` under the given strides. A copy that is a
/// batch of 2-D transposes (see [`transpose_launch`]) goes through the tiled
/// transpose; any other runs one invocation per element.
pub fn wgpu_copy_nd_dispatch(
    input: &DeviceTensor,
    input_offset: usize,
    input_strides: &[isize],
    output: &DeviceTensor,
    output_offset: usize,
    output_shape: &[usize],
    output_strides: &[isize],
) -> TractResult<()> {
    let geo = merge_copy_axes(input_strides, output_shape, output_strides);
    if let Some(t) = transpose_launch(&geo.input_strides, &geo.shape, &geo.output_strides) {
        return wgpu_transpose_dispatch(input, input_offset, output, output_offset, &t);
    }
    ensure!(
        geo.shape.len() <= PARAM_AXES,
        "A copy of {output_shape:?} merges to rank {}, past the {PARAM_AXES} axes the shader takes",
        geo.shape.len()
    );
    let (output_shape, input_strides, output_strides) =
        (&geo.shape[..], &geo.input_strides[..], &geo.output_strides[..]);
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

fn wgpu_transpose_dispatch(
    input: &DeviceTensor,
    input_offset: usize,
    output: &DeviceTensor,
    output_offset: usize,
    t: &TransposeLaunch,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let key = PipelineKey {
            module: ModuleKey { kind: ModuleKind::Transpose, dtype: dt },
            entry: EntryPoint::typed("copy_transpose", dt),
        };
        let pipeline = q.context().pipeline(key)?;
        let in_buf = get_wgpu_buffer(input);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(LayoutKind::Unary, &[in_buf, out_buf], q.uniform())?;
        let [col_tiles, row_tiles, batch] = t.groups;
        let tiles = col_tiles * row_tiles * batch;
        let vals = [
            element_offset(input, input_offset),
            element_offset(output, output_offset),
            t.rows,
            t.cols,
            t.in_batch_stride as usize,
            t.in_col_stride as usize,
            t.out_batch_stride as usize,
            t.out_row_stride as usize,
            col_tiles,
            row_tiles,
            tiles,
        ]
        .map(|v| v as u32);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        let x = tiles.min(GRID_LIMIT as usize) as u32;
        let y = tiles.div_ceil(GRID_LIMIT as usize) as u32;
        q.dispatch_grid("copy_transpose", &pipeline, &bg, dyn_off, [x, y, 1])
    })
}
