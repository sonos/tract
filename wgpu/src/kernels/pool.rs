use tract_core::internal::*;
use tract_core::ops::cnn::{MaxPool, OptMaxPool, OptSumPool, PoolSpec, SumPool};
use tract_gpu::ops::pool::{GpuPool, PoolKind};
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, WORKGROUP,
    keys_for, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Pool, &["max_pool_2d", "sum_pool_2d", "global_avg_pool_2d"], shader_f16)
}

pub fn wgpu_pool_supported(pool_spec: &PoolSpec, fact: &TypedFact) -> bool {
    matches!(fact.datum_type, DatumType::F16 | DatumType::F32)
        && fact.rank() == 4
        && pool_spec.kernel_shape.len() == 2
        && fact.shape.as_concrete().is_some()
}

pub fn wgpu_pool_dispatch(
    pool_spec: &PoolSpec,
    kind: PoolKind,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let in_shape = pool_spec.data_format.shape(input.shape())?;
        let out_shape = pool_spec.data_format.shape(output.shape())?;
        ensure!(in_shape.hw_rank() == 2);
        let strides = pool_spec.strides();
        let dilations = pool_spec.dilations();
        let padding = pool_spec.computed_padding(in_shape.hw_dims());
        let (count_include_pad, normalize) = match kind {
            PoolKind::Max => (0u32, 0u32),
            PoolKind::Sum { count_include_pad, normalize } => {
                (count_include_pad as u32, normalize as u32)
            }
        };
        let channels_last = pool_spec.data_format.c_is_last() as u32;
        let n = *in_shape.n().unwrap_or(&1);
        let c = *in_shape.c();
        let oh = out_shape.hw_dims()[0];
        let ow = out_shape.hw_dims()[1];
        let global_average = matches!(kind, PoolKind::Sum { normalize: true, .. })
            && oh == 1
            && ow == 1
            && padding.iter().all(|p| p.pad_before == 0 && p.pad_after == 0)
            && dilations.as_ref() == [1, 1]
            && pool_spec.kernel_shape.as_slice() == in_shape.hw_dims();
        let stem = if global_average {
            "global_avg_pool_2d"
        } else {
            match kind {
                PoolKind::Max => "max_pool_2d",
                PoolKind::Sum { .. } => "sum_pool_2d",
            }
        };
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::Pool, dtype: dt },
            entry: EntryPoint::typed(stem, dt),
        })?;
        let nelem = (n * oh * ow * c) as u64;
        let in_buf = get_wgpu_buffer(input);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(LayoutKind::Unary, &[in_buf, out_buf], q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            n as u32,
            in_shape.hw_dims()[0] as u32,
            in_shape.hw_dims()[1] as u32,
            c as u32,
            oh as u32,
            ow as u32,
            pool_spec.kernel_shape[0] as u32,
            pool_spec.kernel_shape[1] as u32,
            strides[0] as u32,
            strides[1] as u32,
            padding[0].pad_before as u32,
            padding[1].pad_before as u32,
            dilations[0] as u32,
            dilations[1] as u32,
            count_include_pad,
            normalize,
            channels_last,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        if global_average {
            q.dispatch_grid("pool_global", &pipeline, &bg, dyn_off, [(n * c) as u32, 1, 1])
        } else {
            q.dispatch("pool", &pipeline, &bg, dyn_off, nelem)
        }
    })
}

fn try_make_wgpu_pool(
    source: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<Box<dyn TypedOp>>> {
    let facts = source.node_input_facts(node.id)?;
    rule_if_some!(op = GpuPool::from_core(node.op.as_ref(), "Wgpu", wgpu_pool_dispatch));
    rule_if!(wgpu_pool_supported(&op.pool_spec, facts[0]));
    Ok(Some(Box::new(op)))
}

crate::register_wgpu_op!(MaxPool, |source, node, _op| try_make_wgpu_pool(source, node));
crate::register_wgpu_op!(SumPool, |source, node, _op| try_make_wgpu_pool(source, node));
crate::register_wgpu_op!(OptMaxPool, |source, node, _op| try_make_wgpu_pool(source, node));
crate::register_wgpu_op!(OptSumPool, |source, node, _op| try_make_wgpu_pool(source, node));

/// Referenced from the transform so the registrations above survive linking.
pub fn link_translators() {}

pub(crate) fn pool_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let neg_inf = match dt {
        ShaderDtype::F32 => "f32(-3.402823e+38)",
        ShaderDtype::F16 => "f16(-65504.0)",
    };
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    n: i32,
    ih: i32,
    iw: i32,
    c: i32,
    oh: i32,
    ow: i32,
    kh: i32,
    kw: i32,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
    dil_h: i32,
    dil_w: i32,
    count_include_pad: i32,
    normalize: i32,
    channels_last: i32,
    _p0: i32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> pool_scratch: array<f32, {WORKGROUP}>;

fn in_idx_nhwc(n: i32, h: i32, w: i32, c: i32) -> u32 {{
    return u32(((n * params.ih + h) * params.iw + w) * params.c + c);
}}
fn in_idx_nchw(n: i32, c: i32, h: i32, w: i32) -> u32 {{
    return u32(((n * params.c + c) * params.ih + h) * params.iw + w);
}}
fn out_idx_nhwc(n: i32, h: i32, w: i32, c: i32) -> u32 {{
    return u32(((n * params.oh + h) * params.ow + w) * params.c + c);
}}
fn out_idx_nchw(n: i32, c: i32, h: i32, w: i32) -> u32 {{
    return u32(((n * params.c + c) * params.oh + h) * params.ow + w);
}}

@compute @workgroup_size({WORKGROUP})
fn global_avg_pool_2d_{suf}(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>
) {{
    let nc = i32(group.x);
    if (nc >= params.n * params.c) {{ return; }}
    let c = nc % params.c;
    let n = nc / params.c;
    let spatial = params.ih * params.iw;
    var acc = 0.0;
    for (var i = i32(local.x); i < spatial; i += {WORKGROUP}) {{
        let h = i / params.iw;
        let w = i - h * params.iw;
        var idx: u32;
        if (params.channels_last != 0) {{
            idx = in_idx_nhwc(n, h, w, c);
        }} else {{
            idx = in_idx_nchw(n, c, h, w);
        }}
        acc += f32(inp[params.off_in + idx]);
    }}
    pool_scratch[local.x] = acc;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {{
        if (local.x < stride) {{
            pool_scratch[local.x] += pool_scratch[local.x + stride];
        }}
        workgroupBarrier();
    }}
    if (local.x == 0u) {{
        var oidx: u32;
        if (params.channels_last != 0) {{
            oidx = out_idx_nhwc(n, 0, 0, c);
        }} else {{
            oidx = out_idx_nchw(n, c, 0, 0);
        }}
        outp[params.off_out + oidx] = {t}(pool_scratch[0] / f32(spatial));
    }}
}}

@compute @workgroup_size({WORKGROUP})
fn max_pool_2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = i32(gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x);
    let nout = params.n * params.oh * params.ow * params.c;
    if (i >= nout) {{ return; }}
    var rest = i;
    let c = rest % params.c; rest = rest / params.c;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh;
    let n = rest / params.oh;
    let h0 = oh * params.stride_h - params.pad_h;
    let w0 = ow * params.stride_w - params.pad_w;
    var best = {neg_inf};
    for (var kh = 0; kh < params.kh; kh++) {{
        let ih = h0 + kh * params.dil_h;
        if (ih < 0 || ih >= params.ih) {{ continue; }}
        for (var kw = 0; kw < params.kw; kw++) {{
            let iw = w0 + kw * params.dil_w;
            if (iw < 0 || iw >= params.iw) {{ continue; }}
            var idx: u32;
            if (params.channels_last != 0) {{
                idx = in_idx_nhwc(n, ih, iw, c);
            }} else {{
                idx = in_idx_nchw(n, c, ih, iw);
            }}
            best = max(best, inp[params.off_in + idx]);
        }}
    }}
    var oidx: u32;
    if (params.channels_last != 0) {{
        oidx = out_idx_nhwc(n, oh, ow, c);
    }} else {{
        oidx = out_idx_nchw(n, c, oh, ow);
    }}
    outp[params.off_out + oidx] = best;
}}

@compute @workgroup_size({WORKGROUP})
fn sum_pool_2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = i32(gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x);
    let nout = params.n * params.oh * params.ow * params.c;
    if (i >= nout) {{ return; }}
    var rest = i;
    let c = rest % params.c; rest = rest / params.c;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh;
    let n = rest / params.oh;
    let h0 = oh * params.stride_h - params.pad_h;
    let w0 = ow * params.stride_w - params.pad_w;
    var acc = 0.0;
    var count = 0;
    for (var kh = 0; kh < params.kh; kh++) {{
        let ih = h0 + kh * params.dil_h;
        if (ih < 0 || ih >= params.ih) {{ continue; }}
        for (var kw = 0; kw < params.kw; kw++) {{
            let iw = w0 + kw * params.dil_w;
            if (iw < 0 || iw >= params.iw) {{ continue; }}
            var idx: u32;
            if (params.channels_last != 0) {{
                idx = in_idx_nhwc(n, ih, iw, c);
            }} else {{
                idx = in_idx_nchw(n, c, ih, iw);
            }}
            acc += f32(inp[params.off_in + idx]);
            count += 1;
        }}
    }}
    if (params.normalize != 0) {{
        var denom = count;
        if (params.count_include_pad != 0) {{
            denom = params.kh * params.kw;
        }}
        if (denom > 0) {{
            acc = acc / f32(denom);
        }}
    }}
    var oidx: u32;
    if (params.channels_last != 0) {{
        oidx = out_idx_nhwc(n, oh, ow, c);
    }} else {{
        oidx = out_idx_nchw(n, c, oh, ow);
    }}
    outp[params.off_out + oidx] = {t}(acc);
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;

    use crate::test_utils::{fill, gpu_vs_cpu};

    /// Squeeze/excitation blocks use full-spatial average pools. The WGPU path
    /// reduces each channel across a workgroup instead of making one invocation
    /// walk the complete image serially.
    #[test]
    fn global_average_pool_matches_cpu() -> TractResult<()> {
        use tract_core::ops::cnn::{PaddingSpec, PoolSpec, SumPool};
        use tract_core::ops::nn::DataFormat;
        for (n, c, h, w) in [(1usize, 16usize, 64usize, 64usize), (2, 3, 7, 11)] {
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact([n, c, h, w]))?;
            let pool = SumPool {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![h, w],
                    padding: PaddingSpec::Valid,
                    dilations: None,
                    strides: Some(tvec![h, w]),
                    input_channels: c,
                    output_channels: c,
                },
                count_include_pad: true,
                normalize: true,
            };
            let y = model.wire_node("global-average", pool, &[x])?[0];
            model.select_output_outlets(&[y])?;
            gpu_vs_cpu(model, Tensor::from_shape(&[n, c, h, w], &fill(17, n * c * h * w))?)
                .with_context(|| format!("n{n} c{c} {h}x{w}"))?;
        }
        Ok(())
    }
}
