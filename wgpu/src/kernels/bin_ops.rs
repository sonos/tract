use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    AT8, EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype,
    WORKGROUP, Width, binary_ops_vec4_wgsl, binary_ops_wgsl, broadcast_strides, dtypes, op_in_set,
    pack_u32s, pad8_u32, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    let mut keys = Vec::new();
    for dt in dtypes(shader_f16) {
        for name in BINARY_OPS {
            for width in [Width::Scalar, Width::Vec4, Width::Vec4Splat] {
                keys.push(PipelineKey {
                    module: ModuleKey { kind: ModuleKind::Binary, dtype: *dt },
                    entry: EntryPoint::wide(name, width, *dt),
                });
            }
        }
    }
    keys
}

/// A wider thread only pays when the left operand runs with the output and the
/// right one either does too or is constant across the four values.
fn width_for(lhs: &DeviceTensor, rhs: &DeviceTensor, output: &DeviceTensor) -> Width {
    let out_shape = output.shape();
    let last = *out_shape.last().unwrap_or(&1);
    if !output.len().is_multiple_of(4) || !last.is_multiple_of(4) {
        return Width::Scalar;
    }
    let natural = Tensor::natural_strides(out_shape);
    let runs_with_output = |t: &DeviceTensor| t.shape() == out_shape && t.strides() == &natural[..];
    if !runs_with_output(lhs) {
        return Width::Scalar;
    }
    if runs_with_output(rhs) {
        Width::Vec4
    } else if rhs.rank() == out_shape.len() && rhs.shape().last() == Some(&1) && last != 1 {
        Width::Vec4Splat
    } else {
        Width::Scalar
    }
}

pub fn is_supported(mini_op: &dyn BinMiniOp, dt: DatumType) -> bool {
    let name = mini_op.name().to_lowercase();
    BINARY_OPS.contains(&name.as_str()) && matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn wgpu_bin_op_dispatch(
    mini_op: &dyn BinMiniOp,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(lhs);
        q.retain_tensor(rhs);
        q.retain_tensor(output);
        ensure!(lhs.rank() == rhs.rank());
        let dt = ShaderDtype::from_datum(lhs.datum_type())?;
        let name = op_in_set(BINARY_OPS, mini_op.name())
            .with_context(|| format!("tract-wgpu has no binary kernel for {}", mini_op.name()))?;
        let width = width_for(lhs, rhs, output);
        let key = PipelineKey {
            module: ModuleKey { kind: ModuleKind::Binary, dtype: dt },
            entry: EntryPoint::wide(name, width, dt),
        };
        let pipeline = q.context().pipeline(key)?;
        let lhs_buf = get_wgpu_buffer(lhs);
        let rhs_buf = get_wgpu_buffer(rhs);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(
            LayoutKind::Binary,
            &[lhs_buf, rhs_buf, out_buf],
            q.uniform(),
        )?;

        let out_shape = output.shape();
        let lhs_s = broadcast_strides(lhs.shape(), lhs.strides(), out_shape);
        let rhs_s = broadcast_strides(rhs.shape(), rhs.strides(), out_shape);
        let shp = pad8_u32(out_shape);
        let mut vals = vec![
            element_offset(lhs, 0) as u32,
            element_offset(rhs, 0) as u32,
            element_offset(output, 0) as u32,
            out_shape.len() as u32,
            output.len() as u32,
            0,
            0,
            0,
        ];
        vals.extend_from_slice(&lhs_s);
        vals.extend_from_slice(&rhs_s);
        vals.extend_from_slice(&shp);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        let threads = if width == Width::Scalar { output.len() } else { output.len() / 4 };
        q.dispatch("binary", &pipeline, &bg, dyn_off, threads as u64)
    })
}

pub fn wgpu_bin_op(mini_op: Box<dyn BinMiniOp>) -> tract_gpu::ops::binary::GpuBinOp {
    tract_gpu::ops::binary::GpuBinOp::new(mini_op, "Wgpu", wgpu_bin_op_dispatch)
}

register_wgpu_op!(tract_core::ops::binary::TypedBinOp, |source, node, op| {
    rule_if!(is_supported(&*op.0, source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(wgpu_bin_op(op.0.clone()))))
});

pub(crate) fn binary_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let lib = binary_ops_wgsl(t);
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_lhs: u32,
    off_rhs: u32,
    off_out: u32,
    rank: u32,
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    lhs_s0: vec4<u32>,
    lhs_s1: vec4<u32>,
    rhs_s0: vec4<u32>,
    rhs_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> lhs: array<{t}>;
@group(0) @binding(1) var<storage, read> rhs: array<{t}>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

{AT8}

fn gather_idx(linear: u32, is_lhs: bool) -> u32 {{
    var rest = linear;
    var idx = 0u;
    let r = params.rank;
    for (var k = 0u; k < r; k++) {{
        let axis = r - 1u - k;
        let dim = max(at8(params.out_sh0, params.out_sh1, axis), 1u);
        let c = rest % dim;
        rest = rest / dim;
        let stride = select(
            at8(params.rhs_s0, params.rhs_s1, axis),
            at8(params.lhs_s0, params.lhs_s1, axis),
            is_lhs,
        );
        idx += c * stride;
    }}
    return idx;
}}

{lib}"#
    ));
    for name in BINARY_OPS {
        s.push_str(&format!(
            r#"
@compute @workgroup_size({WORKGROUP})
fn {name}_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let li = params.off_lhs + gather_idx(i, true);
    let ri = params.off_rhs + gather_idx(i, false);
    outp[params.off_out + i] = op_{name}(lhs[li], rhs[ri]);
}}
"#
        ));
    }
    s.push_str(&binary_vec4_entries(t, suf));
    s
}

/// The four-wide binary entries. `_v4` is both operands element for element
/// with the output; `_v4s` is a right operand that broadcasts over the last
/// axis, so one value covers the four a thread handles. Anything else gathers
/// per element and gains nothing from a wider thread.
fn binary_vec4_entries(t: &str, suf: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        r#"
fn lhs4(i: u32) -> vec4<{t}> {{
    let b = params.off_lhs + i * 4u;
    return vec4<{t}>(lhs[b], lhs[b + 1u], lhs[b + 2u], lhs[b + 3u]);
}}

fn rhs4(i: u32) -> vec4<{t}> {{
    let b = params.off_rhs + i * 4u;
    return vec4<{t}>(rhs[b], rhs[b + 1u], rhs[b + 2u], rhs[b + 3u]);
}}

fn store4(i: u32, v: vec4<{t}>) {{
    let b = params.off_out + i * 4u;
    outp[b] = v.x;
    outp[b + 1u] = v.y;
    outp[b + 2u] = v.z;
    outp[b + 3u] = v.w;
}}
"#
    ));
    s.push_str(&binary_ops_vec4_wgsl(t));
    for name in BINARY_OPS {
        s.push_str(&format!(
            r#"
@compute @workgroup_size({WORKGROUP})
fn {name}_v4_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i * 4u >= params.len) {{ return; }}
    store4(i, op_{name}_v(lhs4(i), rhs4(i)));
}}

@compute @workgroup_size({WORKGROUP})
fn {name}_v4s_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i * 4u >= params.len) {{ return; }}
    let r = vec4<{t}>(rhs[params.off_rhs + gather_idx(i * 4u, false)]);
    store4(i, op_{name}_v(lhs4(i), r));
}}
"#
        ));
    }
    s
}

pub const BINARY_OPS: &[&str] = &["mul", "add", "div", "sub", "pow", "min", "max"];
