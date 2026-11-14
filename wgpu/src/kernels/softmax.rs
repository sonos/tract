use tract_core::internal::*;
use tract_core::ops::nn::SoftmaxKind;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, WORKGROUP,
    keys_for, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

const ROW_WG: u32 = 256;
const ROW_MIN_K: usize = 128;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Softmax, &["softmax", "softmax_rows"], shader_f16)
}

fn reshape_to_rank_3(shape: &[usize], axis: usize) -> [usize; 3] {
    let outer: usize = shape[..axis].iter().product();
    let k = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    [outer.max(1), k, inner.max(1)]
}

pub fn wgpu_softmax_dispatch(
    input: &DeviceTensor,
    axis: usize,
    kind: SoftmaxKind,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        ensure!(output.shape() == input.shape());
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let nd3 = reshape_to_rank_3(input.shape(), axis);
        let n = (nd3[0] * nd3[2]) as u64;
        if n == 0 || nd3[1] == 0 {
            return Ok(());
        }
        let rows = nd3[2] == 1 && nd3[1] >= ROW_MIN_K;
        let entry = if rows { "softmax_rows" } else { "softmax" };
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::Softmax, dtype: dt },
            entry: EntryPoint::typed(entry, dt),
        })?;
        let in_buf = get_wgpu_buffer(input);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(LayoutKind::Unary, &[in_buf, out_buf], q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            nd3[0] as u32,
            nd3[1] as u32,
            nd3[2] as u32,
            (kind == SoftmaxKind::LogSoftmax) as u32,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        if rows {
            let x = n.min(GRID_LIMIT as u64) as u32;
            let y = n.div_ceil(GRID_LIMIT as u64) as u32;
            q.dispatch_grid("softmax_rows", &pipeline, &bg, dyn_off, [x, y, 1])
        } else {
            q.dispatch("softmax", &pipeline, &bg, dyn_off, n)
        }
    })
}

register_wgpu_op!(tract_core::ops::nn::Softmax, |source, node, op| {
    rule_if!(matches!(
        source.node_input_facts(node.id)?[0].datum_type,
        DatumType::F32 | DatumType::F16
    ));
    rule_if!(op.quant_output_dt.is_none());
    rule_if!(op.axes.len() == 1);
    Ok(Some(Box::new(tract_gpu::ops::softmax::GpuSoftmax::from_tract_core(
        op,
        "Wgpu",
        wgpu_softmax_dispatch,
    )?)))
});

pub(crate) fn softmax_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    outer: u32,
    k: u32,
    inner: u32,
    log_out: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn softmax_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var m = f32(inp[params.off_in + (o * params.k) * params.inner + r]);
    for (var k = 1u; k < params.k; k++) {{
        m = max(m, f32(inp[params.off_in + (o * params.k + k) * params.inner + r]));
    }}
    var sum = 0.0;
    for (var k = 0u; k < params.k; k++) {{
        sum += exp(f32(inp[params.off_in + (o * params.k + k) * params.inner + r]) - m);
    }}
    let inv = 1.0 / sum;
    let shift = m + log(sum);
    for (var k = 0u; k < params.k; k++) {{
        let x = f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
        var y = exp(x - m) * inv;
        if (params.log_out != 0u) {{ y = x - shift; }}
        outp[params.off_out + (o * params.k + k) * params.inner + r] = {t}(y);
    }}
}}

var<workgroup> part_max: array<f32, {ROW_WG}>;
var<workgroup> part_sum: array<f32, {ROW_WG}>;

fn is_neg_inf(v: f32) -> bool {{ return bitcast<u32>(v) == 0xff800000u; }}

// A running exp-sum taken from its max `m_from` to `m_to`. A sum that has only seen -inf holds
// nothing: exp(-inf - -inf) would be NaN.
fn rescaled(s: f32, m_from: f32, m_to: f32) -> f32 {{
    if (is_neg_inf(m_from)) {{ return 0.0; }}
    return s * exp(m_from - m_to);
}}

@compute @workgroup_size({ROW_WG})
fn softmax_rows_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.outer) {{ return; }}
    let src = params.off_in + row * params.k;
    let dst = params.off_out + row * params.k;
    // A constant -inf is a WGSL shader-creation error in Tint (Chrome); build it at runtime.
    var neg_inf_bits = 0xff800000u;
    var m = bitcast<f32>(neg_inf_bits);
    var s = 0.0;
    for (var i = lid.x; i < params.k; i += {ROW_WG}u) {{
        let x = f32(inp[src + i]);
        if (x > m) {{ s = rescaled(s, m, x); m = x; }}
        if (!is_neg_inf(m)) {{ s += exp(x - m); }}
    }}
    part_max[lid.x] = m;
    part_sum[lid.x] = s;
    for (var stride = {ROW_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            let m0 = part_max[lid.x];
            let m1 = part_max[lid.x + stride];
            let mm = max(m0, m1);
            part_sum[lid.x] = rescaled(part_sum[lid.x], m0, mm) + rescaled(part_sum[lid.x + stride], m1, mm);
            part_max[lid.x] = mm;
        }}
    }}
    workgroupBarrier();
    let mx = part_max[0];
    let sum = part_sum[0];
    let inv = 1.0 / sum;
    let shift = mx + log(sum);
    for (var i = lid.x; i < params.k; i += {ROW_WG}u) {{
        let x = f32(inp[src + i]);
        var y = exp(x - mx) * inv;
        if (params.log_out != 0u) {{ y = x - shift; }}
        outp[dst + i] = {t}(y);
    }}
}}
"#
    ));
    s
}
