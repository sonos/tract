use tract_core::internal::*;
use tract_gpu::ops::reduce::Reducer;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, WORKGROUP,
    keys_for, op_in_set, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// The reducers the module has an entry point for, spelled as the WGSL names
/// them.
const REDUCE_OPS: &[&str] = &[
    "reduce_sum",
    "reduce_prod",
    "reduce_max",
    "reduce_min",
    "reduce_mean",
    "reduce_mean_of_squares",
];

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Reduce, REDUCE_OPS, shader_f16)
}

fn reshape_to_rank_3(shape: &[usize], axis: usize) -> [usize; 3] {
    let outer: usize = shape[..axis].iter().product();
    let k = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    [outer.max(1), k, inner.max(1)]
}

pub fn wgpu_reduce_launch(
    reducer: &Reducer,
    input: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        ensure!(output.datum_type() == input.datum_type());
        ensure!(output.shape()[axis] == 1);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        if reducer.is_logic() {
            bail!("tract-wgpu reduce has no bool path yet");
        }
        let name = op_in_set(REDUCE_OPS, &format!("reduce_{reducer}"))
            .with_context(|| format!("tract-wgpu has no reduce kernel for {reducer}"))?;
        let key = PipelineKey {
            module: ModuleKey { kind: ModuleKind::Reduce, dtype: dt },
            entry: EntryPoint::typed(name, dt),
        };
        let pipeline = q.context().pipeline(key)?;
        let nd3 = reshape_to_rank_3(input.shape(), axis);
        let n = (nd3[0] * nd3[2]) as u64;
        if n == 0 || nd3[1] == 0 {
            return Ok(());
        }
        let in_buf = get_wgpu_buffer(input);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(LayoutKind::Unary, &[in_buf, out_buf], q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            nd3[0] as u32,
            nd3[1] as u32,
            nd3[2] as u32,
            0,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("reduce", &pipeline, &bg, dyn_off, n)
    })
}

/// Whether a floating point sum or FP16 mean covers a trailing run of axes
/// large enough for cooperative reduction.
pub fn is_trailing_reduce_run(
    op: &tract_core::ops::nn::Reduce,
    shape: &[TDim],
    dt: DatumType,
) -> bool {
    use tract_core::ops::nn::Reducer;
    let Some(&first) = op.axes.first().filter(|&&axis| axis < shape.len()) else {
        return false;
    };
    ((op.reducer == Reducer::Sum && dt.is_float())
        || (op.reducer == Reducer::Mean
            && dt == DatumType::F16
            && op.axes.len() == 2
            && shape[first].to_usize().is_ok_and(|rows| (1..=256).contains(&rows))
            && shape[..first].iter().product::<TDim>().to_usize().is_ok_and(|n| n <= 65535)))
        && op.axes.len() > 1
        && op.axes.iter().copied().eq(first..shape.len())
        && shape[first..]
            .iter()
            .product::<TDim>()
            .to_usize()
            .is_ok_and(|k| k >= SUM_RUN_WG as usize / 4)
}

/// One workgroup per output value, its threads striding over the reduced
/// values and meeting in workgroup memory.
pub fn wgpu_sum_run_launch(
    input: &DeviceTensor,
    first_axis: usize,
    mean: bool,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let outer: usize = input.shape()[..first_axis].iter().product();
        let k: usize = input.shape()[first_axis..].iter().product();
        ensure!(outer <= 65535, "sum run over {outer} outputs exceeds one grid dimension");
        let layout = LayoutKind::Unary;
        let (width, height) =
            if mean { (input.shape()[first_axis + 1], input.shape()[first_axis]) } else { (k, 0) };
        let offset = element_offset(input, 0);
        let vec4 = mean
            && dt == ShaderDtype::F16
            && matches!((height, width, outer), (45, 80, 72 | 120) | (23, 40, 480 | 672 | 960))
            && offset.is_multiple_of(4)
            && input.strides() == &Tensor::natural_strides(input.shape())[..];
        let wg = if vec4 && height <= 32 { 32 } else { 64 };
        let name = match (mean, vec4) {
            (true, true) => "mean_run_vec4",
            (true, false) => "mean_run",
            (false, _) => "sum_run",
        };
        let pipeline = q.context().chain_pipeline(
            (name, dt, wg),
            layout,
            EntryPoint::typed(name, dt),
            || {
                if mean { rounded_mean_wgsl(name, vec4, wg) } else { sum_run_module(dt) }
            },
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            offset as u32,
            element_offset(output, 0) as u32,
            width as u32,
            height as u32,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch_grid(name, &pipeline, &bg, dyn_off, [outer as u32, 1, 1])
    })
}

register_wgpu_op!(tract_core::ops::nn::Reduce, |source, node, op| {
    let fact = &source.node_input_facts(node.id)?[0];
    let dt = fact.datum_type;
    if is_trailing_reduce_run(op, fact.shape.dims(), dt) {
        return Ok(Some(Box::new(crate::ops::sum_run::WgpuSumRun {
            first_axis: op.axes[0],
            mean: op.reducer == tract_core::ops::nn::Reducer::Mean,
        })));
    }
    if let Ok(gpu_op) =
        tract_gpu::ops::reduce::GpuReduce::from_tract_core(op, "Wgpu", wgpu_reduce_launch)
    {
        rule_if!(gpu_op.reducer.is_supported_dt(dt));
        return Ok(Some(Box::new(gpu_op)));
    }
    Ok(None)
});

/// Threads in a cooperative sum: each output gets one workgroup this wide.
pub const SUM_RUN_WG: u32 = 256;

/// A sum over trailing values: one workgroup per output,
/// threads striding over its `k` values, then a tree over workgroup memory.
fn sum_run_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    k: u32,
    _p0: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial: array<f32, {SUM_RUN_WG}>;

@compute @workgroup_size({SUM_RUN_WG})
fn sum_run_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let base = params.off_in + wid.x * params.k;
    var acc = 0.0;
    for (var i = lid.x; i < params.k; i += {SUM_RUN_WG}u) {{
        acc += f32(inp[base + i]);
    }}
    partial[lid.x] = acc;
    for (var stride = {SUM_RUN_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] += partial[lid.x + stride];
        }}
    }}
    if (lid.x == 0u) {{
        outp[params.off_out + wid.x] = {t}(partial[0]);
    }}
}}
"#
    ));
    s
}

/// A fused width then height mean. Both sums visit values in their original
/// order and the intermediate row mean is rounded to f16.
fn rounded_mean_wgsl(name: &str, vec4: bool, wg: u32) -> String {
    let (input_type, sum_row) = if vec4 {
        (
            "vec4<f16>",
            r#"
        for (var x = 0u; x < params.width / 4u; x++) {
            let v = inp[row / 4u + x];
            acc += f32(v.x);
            acc += f32(v.y);
            acc += f32(v.z);
            acc += f32(v.w);
        }"#,
        )
    } else {
        (
            "f16",
            r#"
        for (var x = 0u; x < params.width; x++) {
            acc += f32(inp[row + x]);
        }"#,
        )
    };
    format!(
        r#"enable f16;
struct Params {{ off_in: u32, off_out: u32, width: u32, height: u32 }}
@group(0) @binding(0) var<storage, read> inp: array<{input_type}>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> rows: array<f32, 256>;

@compute @workgroup_size({wg})
fn {name}_f16(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    for (var y = lid.x; y < params.height; y += {wg}u) {{
        let row = params.off_in + (wid.x * params.height + y) * params.width;
        var acc = 0.0;
        {sum_row}
        rows[y] = f32(f16(acc / f32(params.width)));
    }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        var acc = 0.0;
        for (var y = 0u; y < params.height; y++) {{
            acc += rows[y];
        }}
        outp[params.off_out + wid.x] = f16(acc / f32(params.height));
    }}
}}
"#
    )
}

pub(crate) fn reduce_wgsl(dt: ShaderDtype) -> String {
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
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn reduce_sum_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = 0.0;
    for (var k = 0u; k < params.k; k++) {{
        acc += f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = {t}(acc);
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_prod_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = 1.0;
    for (var k = 0u; k < params.k; k++) {{
        acc *= f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = {t}(acc);
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_max_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = inp[params.off_in + (o * params.k) * params.inner + r];
    for (var k = 1u; k < params.k; k++) {{
        acc = max(acc, inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = acc;
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_min_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = inp[params.off_in + (o * params.k) * params.inner + r];
    for (var k = 1u; k < params.k; k++) {{
        acc = min(acc, inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = acc;
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_mean_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = 0.0;
    for (var k = 0u; k < params.k; k++) {{
        acc += f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = {t}(acc / f32(params.k));
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_mean_of_squares_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = 0.0;
    for (var k = 0u; k < params.k; k++) {{
        let v = f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
        acc += v * v;
    }}
    outp[params.off_out + i] = {t}(acc / f32(params.k));
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;
    use tract_gpu::tensor::IntoDevice;

    use crate::test_utils::{f16_vs_f32, fill, gpu_vs_cpu};

    /// A sum over both trailing axes at once, as a global pooling is: one
    /// cooperative launch instead of one per axis.
    #[test]
    fn trailing_sum_run_matches_cpu() -> TractResult<()> {
        use tract_core::ops::nn::{Reduce, Reducer};
        for shape in [[3, 20, 17], [16, 9, 16], [2, 1000, 3]] {
            let mut model = TypedModel::default();
            let a = model.add_source("a", f32::fact(shape))?;
            let y = model.wire_node(
                "sum",
                Reduce { axes: tvec![1, 2], reducer: Reducer::Sum },
                &[a],
            )?[0];
            model.select_output_outlets(&[y])?;
            let n: usize = shape.iter().product();
            gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(7, n))?)?;
        }
        Ok(())
    }

    #[test]
    fn fp16_trailing_mean_run_matches_cpu() -> TractResult<()> {
        use tract_core::ops::nn::{Reduce, Reducer};
        let shape = [2, 4, 23, 40];
        let build = |dt: DatumType| -> TractResult<TypedModel> {
            let mut model = TypedModel::default();
            let a = model.add_source("a", dt.fact(shape))?;
            let y = model.wire_node(
                "mean",
                Reduce { axes: tvec![2, 3], reducer: Reducer::Mean },
                &[a],
            )?;
            model.select_output_outlets(&y)?;
            Ok(model)
        };
        let data = Tensor::from_shape(&shape, &fill(13, shape.iter().product()))?;
        f16_vs_f32(build, data, Approximation::Close)
    }

    #[test]
    fn fp16_trailing_mean_preserves_two_pass_rounding() -> TractResult<()> {
        use super::{wgpu_reduce_launch, wgpu_sum_run_launch};
        use tract_gpu::ops::reduce::Reducer;
        crate::wgpu_context();
        for shape @ [n, c, h, _] in [[1, 480, 23, 40], [1, 72, 45, 80], [1, 24, 65, 19]] {
            let input = Tensor::from_shape(&shape, &fill(11, shape.iter().product()))?
                .cast_to::<f16>()?
                .into_owned()
                .into_device()?;
            let rows = Tensor::zero::<f16>(&[n, c, h, 1])?.into_device()?;
            let split = Tensor::zero::<f16>(&[n, c, 1, 1])?.into_device()?;
            let fused = Tensor::zero::<f16>(&[n, c, 1, 1])?.into_device()?;
            wgpu_reduce_launch(&Reducer::Mean, &input, 3, &rows)?;
            wgpu_reduce_launch(&Reducer::Mean, &rows, 2, &split)?;
            wgpu_sum_run_launch(&input, 2, true, &fused)?;
            // WGSL f32 division is not correctly rounded: kernels may differ by an f16 ulp.
            fused
                .to_host()?
                .into_tensor()
                .close_enough(&split.to_host()?.into_tensor(), Approximation::Close)?;
        }
        Ok(())
    }
}
