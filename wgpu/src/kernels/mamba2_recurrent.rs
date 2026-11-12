use tract_core::internal::*;
use tract_gpu::ops::mamba2_recurrent::GpuMamba2Recurrent;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::mamba2_recurrent::Mamba2Recurrent;

use crate::kernels::shaders::{EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, pack_u32s};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

const THREADS: usize = 64;
const PER_THREAD: usize = 4;
const MAX_STATE: usize = THREADS * PER_THREAD;

#[allow(clippy::too_many_arguments)]
pub fn wgpu_mamba2_recurrent_dispatch(
    x: &DeviceTensor,
    b: &DeviceTensor,
    c: &DeviceTensor,
    log_decay: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    let &[batch, steps, heads, head_dim] = x.shape() else { bail!("Mamba2 x must be rank 4") };
    let &[_, _, groups, d_state] = b.shape() else { bail!("Mamba2 b must be rank 4") };
    ensure!(b.datum_type() == x.datum_type() && c.datum_type() == x.datum_type());
    ensure!(b.shape() == c.shape() && groups > 0 && heads % groups == 0);
    ensure!(d_state <= MAX_STATE, "wgpu Mamba2 takes a state of at most {MAX_STATE}");
    ensure!(log_decay.len() == batch * steps * heads);
    ensure!(initial_state.shape() == [batch, heads, head_dim, d_state]);
    let rows = batch * heads * head_dim;
    if rows == 0 {
        return Ok(());
    }
    let tensors = [x, b, c, log_decay, initial_state, output, final_state];
    with_wgpu_queue(|q| {
        for t in tensors {
            q.retain_tensor(t);
        }
        let dts = (
            ShaderDtype::from_datum(x.datum_type())?,
            ShaderDtype::from_datum(log_decay.datum_type())?,
            ShaderDtype::from_datum(initial_state.datum_type())?,
        );
        let layout = LayoutKind::TailWrites { n: 7, writes: 2 };
        let pipeline = q.context().chain_pipeline(
            ("mamba2_recurrent", dts),
            layout,
            EntryPoint::plain("mamba2_recurrent"),
            || mamba2_module(dts.0, dts.1, dts.2),
        )?;
        let buffers: Vec<_> = tensors.iter().map(|t| get_wgpu_buffer(t)).collect();
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let mut params: Vec<u32> = tensors.iter().map(|t| element_offset(t, 0) as u32).collect();
        params.extend([batch, steps, heads, head_dim, groups, d_state].map(|d| d as u32));
        params.extend([0; 3]);
        let dyn_off = q.alloc_uniform(&pack_u32s(&params))?;
        let grid = GRID_LIMIT as usize;
        let groups_xy = [rows.min(grid) as u32, rows.div_ceil(grid) as u32, 1];
        q.dispatch_grid("mamba2_recurrent", &pipeline, &bg, dyn_off, groups_xy)
    })
}

register_wgpu_op!(Mamba2Recurrent, |source, node, _op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[1].datum_type == facts[0].datum_type);
    rule_if!(facts[2].datum_type == facts[0].datum_type);
    rule_if_some!(d_state = facts[1].shape.last().and_then(|d| d.as_i64()));
    rule_if!(d_state as usize <= MAX_STATE);
    Ok(Some(Box::new(GpuMamba2Recurrent {
        backend_name: "Wgpu",
        dispatch: wgpu_mamba2_recurrent_dispatch,
    })))
});

/// One workgroup per (batch, head, p) state row: each thread holds up to
/// [`PER_THREAD`] of the row's values across the S steps, and each step's
/// output is a workgroup tree reduction.
fn mamba2_module(t: ShaderDtype, d: ShaderDtype, s: ShaderDtype) -> String {
    let enable = if [t, d, s].contains(&ShaderDtype::F16) { "enable f16;\n" } else { "" };
    let (t, d, s) = (t.wgsl(), d.wgsl(), s.wgsl());
    format!(
        r#"{enable}
struct Params {{
    off_x: u32,
    off_b: u32,
    off_c: u32,
    off_decay: u32,
    off_state: u32,
    off_out: u32,
    off_final: u32,
    batch: u32,
    steps: u32,
    heads: u32,
    head_dim: u32,
    groups: u32,
    d_state: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> x: array<{t}>;
@group(0) @binding(1) var<storage, read> b: array<{t}>;
@group(0) @binding(2) var<storage, read> c: array<{t}>;
@group(0) @binding(3) var<storage, read> log_decay: array<{d}>;
@group(0) @binding(4) var<storage, read> initial_state: array<{s}>;
@group(0) @binding(5) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(6) var<storage, read_write> final_state: array<{s}>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> partial: array<f32, {THREADS}>;

@compute @workgroup_size({THREADS})
fn mamba2_recurrent(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.batch * params.heads * params.head_dim) {{ return; }}
    let th = lid.x;
    let p = row % params.head_dim;
    let h = (row / params.head_dim) % params.heads;
    let bi = row / (params.head_dim * params.heads);
    let g = h / (params.heads / params.groups);
    let state_base = row * params.d_state;
    var st: array<f32, {PER_THREAD}>;
    for (var k = 0u; k < {PER_THREAD}u; k++) {{
        let n = th + k * {THREADS}u;
        st[k] = 0.0;
        if (n < params.d_state) {{ st[k] = f32(initial_state[params.off_state + state_base + n]); }}
    }}
    for (var t = 0u; t < params.steps; t++) {{
        let step = bi * params.steps + t;
        let xi = (step * params.heads + h) * params.head_dim + p;
        let bc = (step * params.groups + g) * params.d_state;
        let decay = exp(f32(log_decay[params.off_decay + step * params.heads + h]));
        let xv = f32(x[params.off_x + xi]);
        var acc = 0.0;
        for (var k = 0u; k < {PER_THREAD}u; k++) {{
            let n = th + k * {THREADS}u;
            if (n < params.d_state) {{
                st[k] = st[k] * decay + xv * f32(b[params.off_b + bc + n]);
                acc += st[k] * f32(c[params.off_c + bc + n]);
            }}
        }}
        partial[th] = acc;
        workgroupBarrier();
        for (var w = {half}u; w > 0u; w = w >> 1u) {{
            if (th < w) {{ partial[th] += partial[th + w]; }}
            workgroupBarrier();
        }}
        if (th == 0u) {{ outp[params.off_out + xi] = {t}(partial[0]); }}
        workgroupBarrier();
    }}
    for (var k = 0u; k < {PER_THREAD}u; k++) {{
        let n = th + k * {THREADS}u;
        if (n < params.d_state) {{ final_state[params.off_final + state_base + n] = {s}(st[k]); }}
    }}
}}
"#,
        half = THREADS / 2,
    )
}
