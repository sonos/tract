use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LAYER_NORM_WG, LayoutKind, ShaderDtype, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

const MAX_RANK: usize = 5;

fn left_padded<T: Copy>(dims: &[T], fill: T) -> [T; MAX_RANK] {
    let mut padded = [fill; MAX_RANK];
    padded[MAX_RANK - dims.len()..].copy_from_slice(dims);
    padded
}

/// The strides `t` is read through over `shape`, zero along the axes it
/// broadcasts.
fn broadcast_strides(t: &DeviceTensor, shape: &[usize; MAX_RANK]) -> TractResult<[u32; MAX_RANK]> {
    let t_shape = left_padded(t.shape(), 1);
    let t_strides = left_padded(t.strides(), 0);
    let mut strides = [0; MAX_RANK];
    for axis in 0..MAX_RANK {
        if t_shape[axis] == shape[axis] {
            ensure!(t_strides[axis] >= 0, "negative stride in {:?}", t.strides());
            strides[axis] = t_strides[axis] as u32;
        } else {
            ensure!(t_shape[axis] == 1, "{:?} does not broadcast to {shape:?}", t.shape());
        }
    }
    Ok(strides)
}

/// Dispatch for `GpuScaledMaskedSoftmax` with a float mask of the input's
/// dtype. A bool mask never reaches it: the backend keeps bool tensors on the
/// CPU.
pub fn wgpu_scaled_masked_softmax_dispatch(
    input: &DeviceTensor,
    scale: &Tensor,
    mask: &DeviceTensor,
    post_softmax_mask: bool,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    ensure!(mask.datum_type() == input.datum_type(), "wgpu takes float masks only");
    ensure!(!post_softmax_mask, "post_softmax_mask needs a bool mask");
    ensure!((2..=MAX_RANK).contains(&input.rank()) && mask.rank() == input.rank());
    let shape = left_padded(input.shape(), 1);
    let x_strides = broadcast_strides(input, &shape)?;
    let mask_strides = broadcast_strides(mask, &shape)?;
    let k = shape[MAX_RANK - 1];
    let rows = input.len() / k.max(1);
    if rows == 0 || k == 0 {
        return Ok(());
    }
    let scale = scale.cast_to_scalar::<f32>()?;
    with_wgpu_queue(|q| {
        for t in [input, mask, output] {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(3);
        let pipeline = q.context().chain_pipeline(
            ("scaled_masked_softmax", dt),
            layout,
            EntryPoint::typed("scaled_masked_softmax", dt),
            || scaled_masked_softmax_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(mask), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let mut params = vec![
            element_offset(input, 0) as u32,
            element_offset(mask, 0) as u32,
            element_offset(output, 0) as u32,
            shape[1] as u32,
            shape[2] as u32,
            shape[3] as u32,
            k as u32,
            rows as u32,
            scale.to_bits(),
            x_strides[4],
            mask_strides[4],
            0,
        ];
        params.extend(&x_strides[..4]);
        params.extend(&mask_strides[..4]);
        let dyn_off = q.alloc_uniform(&pack_u32s(&params))?;
        let x = rows.min(GRID_LIMIT as usize) as u32;
        let y = rows.div_ceil(GRID_LIMIT as usize) as u32;
        q.dispatch_grid("scaled_masked_softmax", &pipeline, &bg, dyn_off, [x, y, 1])
    })
}

register_wgpu_op!(
    tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax,
    |source, node, op| {
        let facts = source.node_input_facts(node.id)?;
        rule_if!(!op.post_softmax_mask);
        rule_if!(facts[1].datum_type == facts[0].datum_type);
        rule_if!((2..=MAX_RANK).contains(&facts[0].rank()) && facts[1].rank() == facts[0].rank());
        Ok(Some(Box::new(tract_gpu::ops::scaled_masked_softmax::GpuScaledMaskedSoftmax::new(
            op.scale.clone(),
            false,
            "Wgpu",
            wgpu_scaled_masked_softmax_dispatch,
        ))))
    }
);

/// `softmax(x * scale + mask)` over the innermost axis of a rank-5 view, one
/// workgroup per row. The input and mask are read through their strides (the
/// mask's zero along the axes it broadcasts); the output is contiguous.
pub fn scaled_masked_softmax_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_mask: u32,
    off_out: u32,
    d1: u32,
    d2: u32,
    d3: u32,
    k: u32,
    rows: u32,
    scale: f32,
    x_inner: u32,
    mask_inner: u32,
    _pad: u32,
    x_strides: vec4<u32>,
    mask_strides: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> mask: array<{t}>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, {LAYER_NORM_WG}>;

fn logit(x: u32, m: u32, i: u32) -> f32 {{
    return f32(inp[x + i * params.x_inner]) * params.scale + f32(mask[m + i * params.mask_inner]);
}}

@compute @workgroup_size({LAYER_NORM_WG})
fn scaled_masked_softmax_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.rows) {{ return; }}
    let at = vec4<u32>(
        row / (params.d1 * params.d2 * params.d3),
        (row / (params.d2 * params.d3)) % params.d1,
        (row / params.d3) % params.d2,
        row % params.d3,
    );
    let x = params.off_in + dot(at, params.x_strides);
    let m = params.off_mask + dot(at, params.mask_strides);
    let o = params.off_out + row * params.k;
    var hi = -3.4028235e38;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        hi = max(hi, logit(x, m, i));
    }}
    partial[lid.x] = hi;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] = max(partial[lid.x], partial[lid.x + stride]);
        }}
    }}
    workgroupBarrier();
    hi = partial[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        sum += exp(logit(x, m, i) - hi);
    }}
    partial[lid.x] = sum;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] += partial[lid.x + stride];
        }}
    }}
    workgroupBarrier();
    let inv = 1.0 / partial[0];
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        outp[o + i] = {t}(exp(logit(x, m, i) - hi) * inv);
    }}
}}
"#
    ));
    s
}
