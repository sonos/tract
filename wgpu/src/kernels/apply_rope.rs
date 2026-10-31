use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

const MAX_RANK: usize = 4;

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

pub fn wgpu_apply_rope_dispatch(
    input: &DeviceTensor,
    cos: &DeviceTensor,
    sin: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    ensure!(cos.datum_type() == input.datum_type() && sin.datum_type() == input.datum_type());
    ensure!(input.rank() <= MAX_RANK && cos.rank() <= input.rank() && sin.rank() <= input.rank());
    let shape = left_padded(input.shape(), 1);
    ensure!(
        shape[MAX_RANK - 1].is_multiple_of(2),
        "ApplyRope needs an even innermost axis: {shape:?}"
    );
    let x_strides = broadcast_strides(input, &shape)?;
    let cos_strides = broadcast_strides(cos, &shape)?;
    let sin_strides = broadcast_strides(sin, &shape)?;
    let half = shape[MAX_RANK - 1] / 2;
    let len = input.len() / 2;
    if len == 0 {
        return Ok(());
    }
    with_wgpu_queue(|q| {
        for t in [input, cos, sin, output] {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(4);
        let pipeline = q.context().chain_pipeline(
            ("apply_rope", dt),
            layout,
            EntryPoint::typed("apply_rope", dt),
            || apply_rope_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(input),
                get_wgpu_buffer(cos),
                get_wgpu_buffer(sin),
                get_wgpu_buffer(output),
            ],
            q.uniform(),
        )?;
        let mut params = vec![
            element_offset(input, 0) as u32,
            element_offset(cos, 0) as u32,
            element_offset(sin, 0) as u32,
            element_offset(output, 0) as u32,
            shape[1] as u32,
            shape[2] as u32,
            half as u32,
            len as u32,
        ];
        params.extend(x_strides);
        params.extend(cos_strides);
        params.extend(sin_strides);
        let dyn_off = q.alloc_uniform(&pack_u32s(&params))?;
        q.dispatch("apply_rope", &pipeline, &bg, dyn_off, len as u64)
    })
}

pub fn wgpu_rotate_half_dispatch(input: &DeviceTensor, output: &DeviceTensor) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    ensure!(input.rank() > 0);
    let inner = input.shape()[input.rank() - 1];
    ensure!(
        inner.is_multiple_of(2),
        "RotateHalf needs an even innermost axis: {:?}",
        input.shape()
    );
    let len = input.len() / 2;
    if len == 0 {
        return Ok(());
    }
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            ("rotate_half", dt),
            layout,
            EntryPoint::typed("rotate_half", dt),
            || rotate_half_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            (inner / 2) as u32,
            len as u32,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("rotate_half", &pipeline, &bg, dyn_off, len as u64)
    })
}

register_wgpu_op!(tract_transformers::ops::apply_rope::ApplyRope, |source, node, _op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts.iter().all(|f| f.datum_type == facts[0].datum_type));
    rule_if!(facts[0].rank() <= MAX_RANK && facts.iter().all(|f| f.rank() <= facts[0].rank()));
    Ok(Some(Box::new(tract_gpu::ops::apply_rope::GpuApplyRope::new(
        "Wgpu",
        wgpu_apply_rope_dispatch,
    ))))
});

register_wgpu_op!(tract_transformers::ops::apply_rope::RotateHalf, |_source, _node, _op| {
    Ok(Some(Box::new(tract_gpu::ops::rotate_half::GpuRotateHalf::new(
        "Wgpu",
        wgpu_rotate_half_dispatch,
    ))))
});

/// Rotary embedding `x * cos + rotate_half(x) * sin` over a rank-4 view of
/// the input, one thread per pair of elements half an innermost row apart.
/// The input, `cos` and `sin` are read through their strides (zero along the
/// axes `cos` and `sin` broadcast); the output is contiguous.
pub fn apply_rope_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_cos: u32,
    off_sin: u32,
    off_out: u32,
    d1: u32,
    d2: u32,
    half: u32,
    len: u32,
    x_strides: vec4<u32>,
    cos_strides: vec4<u32>,
    sin_strides: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> cosv: array<{t}>;
@group(0) @binding(2) var<storage, read> sinv: array<{t}>;
@group(0) @binding(3) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn apply_rope_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let j = i % params.half;
    let row = i / params.half;
    let i2 = row % params.d2;
    let i1 = (row / params.d2) % params.d1;
    let i0 = row / (params.d2 * params.d1);
    let at = vec4<u32>(i0, i1, i2, j);
    let x = params.off_in + dot(at, params.x_strides);
    let c = params.off_cos + dot(at, params.cos_strides);
    let s = params.off_sin + dot(at, params.sin_strides);
    let x_rot = x + params.half * params.x_strides.w;
    let c_rot = c + params.half * params.cos_strides.w;
    let s_rot = s + params.half * params.sin_strides.w;
    let o = params.off_out + row * 2u * params.half + j;
    outp[o] = inp[x] * cosv[c] - inp[x_rot] * sinv[s];
    outp[o + params.half] = inp[x_rot] * cosv[c_rot] + inp[x] * sinv[s_rot];
}}
"#
    ));
    s
}

/// `concat(-x[half..], x[..half])` along the innermost axis, one thread per
/// pair of elements half a row apart.
pub fn rotate_half_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    half: u32,
    len: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn rotate_half_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let at = (i / params.half) * 2u * params.half + i % params.half;
    let x = inp[params.off_in + at];
    outp[params.off_out + at] = -inp[params.off_in + at + params.half];
    outp[params.off_out + at + params.half] = x;
}}
"#
    ));
    s
}
