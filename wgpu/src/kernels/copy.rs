use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::TransposeLaunch;
use tract_gpu::utils::merge_copy_axes;
use tract_gpu::utils::transpose_launch;

use crate::kernels::shaders::{
    AT8, EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype,
    WORKGROUP, keys_for, pack_u32s, pad8_stride, pad8_u32, preamble,
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

pub(crate) fn copy_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    rank: u32,
    len: u32,
    in_s0: vec4<u32>,
    in_s1: vec4<u32>,
    out_s0: vec4<u32>,
    out_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

{AT8}

@compute @workgroup_size({WORKGROUP})
fn copy_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    var rest = i;
    var in_i = params.off_in;
    var out_i = params.off_out;
    let r = params.rank;
    for (var k = 0u; k < r; k++) {{
        let axis = r - 1u - k;
        let dim = max(at8(params.out_sh0, params.out_sh1, axis), 1u);
        let c = rest % dim;
        rest = rest / dim;
        in_i += c * at8(params.in_s0, params.in_s1, axis);
        out_i += c * at8(params.out_s0, params.out_s1, axis);
    }}
    outp[out_i] = inp[in_i];
}}
"#
    ));
    s
}

/// A batch of 2-D transposes as `tract_gpu::utils::TransposeLaunch` describes
/// them: each workgroup stages one `TRANSPOSE_TILE` square through workgroup
/// memory, reading it along the input's unit-stride rows and writing it along
/// the output's unit-stride columns. Tiles are numbered columns first, then
/// rows, then batch, and the numbering folds onto the grid's x and y.
pub(crate) fn transpose_wgsl(dt: ShaderDtype) -> String {
    use tract_gpu::utils::{TRANSPOSE_ROWS, TRANSPOSE_TILE};
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    rows: u32,
    cols: u32,
    in_batch_stride: u32,
    in_col_stride: u32,
    out_batch_stride: u32,
    out_row_stride: u32,
    col_tiles: u32,
    row_tiles: u32,
    tiles: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<array<{t}, {TRANSPOSE_TILE_PADDED}>, {TRANSPOSE_TILE}>;

@compute @workgroup_size({TRANSPOSE_TILE}, {TRANSPOSE_ROWS})
fn copy_transpose_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = wid.y * {GRID_LIMIT}u + wid.x;
    if (t >= params.tiles) {{ return; }}
    let c0 = (t % params.col_tiles) * {TRANSPOSE_TILE}u;
    let r0 = (t / params.col_tiles % params.row_tiles) * {TRANSPOSE_TILE}u;
    let b = t / params.col_tiles / params.row_tiles;
    let base_in = params.off_in + b * params.in_batch_stride;
    let base_out = params.off_out + b * params.out_batch_stride;
    for (var j = lid.y; j < {TRANSPOSE_TILE}u; j += {TRANSPOSE_ROWS}u) {{
        let r = r0 + lid.x;
        let c = c0 + j;
        if (r < params.rows && c < params.cols) {{
            tile[j][lid.x] = inp[base_in + r + c * params.in_col_stride];
        }}
    }}
    workgroupBarrier();
    for (var j = lid.y; j < {TRANSPOSE_TILE}u; j += {TRANSPOSE_ROWS}u) {{
        let r = r0 + j;
        let c = c0 + lid.x;
        if (r < params.rows && c < params.cols) {{
            outp[base_out + r * params.out_row_stride + c] = tile[lid.x][j];
        }}
    }}
}}
"#,
        TRANSPOSE_TILE_PADDED = TRANSPOSE_TILE + 1,
    ));
    s
}
