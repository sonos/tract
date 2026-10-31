use tract_core::internal::*;
use tract_gpu::ops::group_norm::GroupNormView;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn wgpu_group_norm_stats_dispatch(
    input: &DeviceTensor,
    view: &GroupNormView,
    partials: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(partials);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let (stem, groups) = if view.channel_last {
            let blocks = view.outer * view.channels.div_ceil(GROUP_NORM_WG as usize);
            (EntryPoint::typed("group_norm_stats_channel_last", dt), blocks)
        } else {
            (EntryPoint::typed("group_norm_stats", dt), view.outer * view.groups)
        };
        ensure!(groups <= 65535, "group norm over {groups} workgroups exceeds one grid dimension");
        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            ("group_norm_stats", dt, view.channel_last),
            layout,
            stem,
            || group_norm_stats_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(partials)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(partials, 0) as u32,
            view.pre as u32,
            view.channels as u32,
            view.post as u32,
            view.groups as u32,
            view.splits as u32,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch_grid(
            "group_norm_stats",
            &pipeline,
            &bg,
            dyn_off,
            [groups as u32, view.splits as u32, 1],
        )
    })
}

/// Merges the partials into a `(mean, inverse std)` pair per group, then
/// normalizes the input with them.
pub fn wgpu_group_norm_apply_dispatch(
    input: &DeviceTensor,
    partials: &DeviceTensor,
    view: &GroupNormView,
    eps: f32,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(partials);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let groups = view.outer * view.groups;
        ensure!(groups <= 65535, "group norm over {groups} groups exceeds one grid dimension");
        let stats = DeviceTensor::uninitialized_dt(DatumType::F32, &[groups, 2])?;
        q.retain_tensor(&stats);

        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            "group_norm_finalize",
            layout,
            EntryPoint::plain("group_norm_finalize"),
            group_norm_finalize_module,
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(partials), get_wgpu_buffer(&stats)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(partials, 0) as u32,
            element_offset(&stats, 0) as u32,
            view.entries as u32,
            eps.to_bits(),
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch_grid("group_norm_finalize", &pipeline, &bg, dyn_off, [groups as u32, 1, 1])?;

        let layout = LayoutKind::Binary;
        let pipeline = q.context().chain_pipeline(
            ("group_norm_apply", dt),
            layout,
            EntryPoint::typed("group_norm_apply", dt),
            || group_norm_apply_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(&stats), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(&stats, 0) as u32,
            element_offset(output, 0) as u32,
            input.len() as u32,
            view.pre as u32,
            view.channels as u32,
            view.post as u32,
            view.groups as u32,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("group_norm_apply", &pipeline, &bg, dyn_off, input.len() as u64)
    })
}

/// Threads of each group norm statistics and finalize workgroup.
pub const GROUP_NORM_WG: u32 = 256;

/// Chan's merge of `(count, mean, m2)` triples, and a tree of it over
/// workgroup memory whose result every thread reads.
fn moments_lib() -> String {
    format!(
        r#"
struct Moments {{
    n: f32,
    mean: f32,
    m2: f32,
}}

fn merge_moments(a: Moments, b: Moments) -> Moments {{
    let n = a.n + b.n;
    if (n == 0.0) {{ return a; }}
    let d = b.mean - a.mean;
    let f = b.n / n;
    return Moments(n, a.mean + d * f, a.m2 + b.m2 + d * d * a.n * f);
}}

fn shifted_moments(n: f32, shift: f32, s1: f32, s2: f32) -> Moments {{
    if (n == 0.0) {{ return Moments(0.0, 0.0, 0.0); }}
    return Moments(n, shift + s1 / n, s2 - s1 * s1 / n);
}}

var<workgroup> wg_moments: array<Moments, {GROUP_NORM_WG}>;

fn workgroup_merge_moments(lid: u32, m: Moments) -> Moments {{
    wg_moments[lid] = m;
    for (var stride = {GROUP_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid < stride) {{
            wg_moments[lid] = merge_moments(wg_moments[lid], wg_moments[lid + stride]);
        }}
    }}
    workgroupBarrier();
    return wg_moments[0];
}}
"#
    )
}

/// Partial `(count, mean, m2)` statistics of a group norm input, see
/// `tract_gpu::ops::group_norm::GroupNormParams`. Channel-first, workgroup `x`
/// is a group and `y` a chunk of it; channel-last, each thread walks one
/// channel over a chunk of the positions.
pub fn group_norm_stats_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&moments_lib());
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_partials: u32,
    pre: u32,
    channels: u32,
    post: u32,
    groups: u32,
    splits: u32,
    _p0: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn store_moments(at: u32, m: Moments) {{
    partials[params.off_partials + at * 3u] = m.n;
    partials[params.off_partials + at * 3u + 1u] = m.mean;
    partials[params.off_partials + at * 3u + 2u] = m.m2;
}}

@compute @workgroup_size({GROUP_NORM_WG})
fn group_norm_stats_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let o = wid.x / params.groups;
    let g = wid.x % params.groups;
    let cg = params.channels / params.groups;
    let len = params.pre * cg * params.post;
    let chunk = (len + params.splits - 1u) / params.splits;
    let start = wid.y * chunk;
    let end = min(len, start + chunk);
    var shift = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    var n = 0.0;
    for (var j = start + lid.x; j < end; j += {GROUP_NORM_WG}u) {{
        let q = j % params.post;
        let t = j / params.post;
        let v = f32(inp[params.off_in + ((o * params.pre + t / cg) * params.channels + g * cg + t % cg) * params.post + q]);
        if (n == 0.0) {{ shift = v; }}
        s1 += v - shift;
        s2 += (v - shift) * (v - shift);
        n += 1.0;
    }}
    let m = workgroup_merge_moments(lid.x, shifted_moments(n, shift, s1, s2));
    if (lid.x == 0u) {{
        store_moments(wid.x * params.splits + wid.y, m);
    }}
}}

@compute @workgroup_size({GROUP_NORM_WG})
fn group_norm_stats_channel_last_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let channel_blocks = (params.channels + {GROUP_NORM_WG}u - 1u) / {GROUP_NORM_WG}u;
    let o = wid.x / channel_blocks;
    let c = (wid.x % channel_blocks) * {GROUP_NORM_WG}u + lid.x;
    if (c >= params.channels) {{ return; }}
    let chunk = (params.pre + params.splits - 1u) / params.splits;
    let start = wid.y * chunk;
    let end = min(params.pre, start + chunk);
    var shift = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    var n = 0.0;
    for (var p = start; p < end; p++) {{
        let v = f32(inp[params.off_in + (o * params.pre + p) * params.channels + c]);
        if (n == 0.0) {{ shift = v; }}
        s1 += v - shift;
        s2 += (v - shift) * (v - shift);
        n += 1.0;
    }}
    store_moments((o * params.channels + c) * params.splits + wid.y, shifted_moments(n, shift, s1, s2));
}}
"#
    ));
    s
}

/// Merges each group's partial statistics into its mean and inverse standard
/// deviation, one workgroup per group.
pub fn group_norm_finalize_module() -> String {
    let mut s = moments_lib();
    s.push_str(
        r#"
struct Params {
    off_partials: u32,
    off_stats: u32,
    entries: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

"#,
    );
    s.push_str(&format!(
        r#"
@compute @workgroup_size({GROUP_NORM_WG})
fn group_norm_finalize(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let base = params.off_partials + wid.x * params.entries * 3u;
    var m = Moments(0.0, 0.0, 0.0);
    for (var e = lid.x; e < params.entries; e += {GROUP_NORM_WG}u) {{
        m = merge_moments(m, Moments(partials[base + 3u * e], partials[base + 3u * e + 1u], partials[base + 3u * e + 2u]));
    }}
    m = workgroup_merge_moments(lid.x, m);
    if (lid.x == 0u) {{
        stats[params.off_stats + wid.x * 2u] = m.mean;
        stats[params.off_stats + wid.x * 2u + 1u] = inverseSqrt(m.m2 / m.n + params.eps);
    }}
}}
"#
    ));
    s
}

/// Normalizes each element with its group's `(mean, inverse std)`.
pub fn group_norm_apply_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_stats: u32,
    off_out: u32,
    len: u32,
    pre: u32,
    channels: u32,
    post: u32,
    groups: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn group_norm_apply_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let c = (i / params.post) % params.channels;
    let o = i / (params.pre * params.channels * params.post);
    let at = params.off_stats + (o * params.groups + c / (params.channels / params.groups)) * 2u;
    outp[params.off_out + i] = {t}((f32(inp[params.off_in + i]) - stats[at]) * stats[at + 1u]);
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;

    use crate::test_utils::transformed;

    #[test]
    fn group_norm_runs_on_device() -> TractResult<()> {
        use tract_core::ops::nn::GroupNorm;
        for (shape, axis, reduced_axes) in
            [([2, 32, 9, 7], 1, tvec![2, 3]), ([2, 9, 7, 32], 3, tvec![1, 2])]
        {
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact(shape))?;
            let op = GroupNorm { axis, groups: 8, reduced_axes, eps: rctensor0(1e-5f32) };
            let y = model.wire_node("group_norm", op, &[x])?[0];
            model.select_output_outlets(&[y])?;
            crate::ensure_wgpu_coverage(&transformed(model)?)?;
        }
        Ok(())
    }
}
