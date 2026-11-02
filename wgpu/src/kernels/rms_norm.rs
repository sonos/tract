use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LAYER_NORM_WG, LayoutKind, ShaderDtype, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Dispatch for `GpuRmsNorm`. `residual` and `sum_out` come and go together:
/// the kernel adds them in the input dtype, writes that sum, and normalizes it.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_rms_norm_dispatch(
    input: &DeviceTensor,
    residual: Option<&DeviceTensor>,
    scale: Option<&DeviceTensor>,
    round_scale_f16: bool,
    axis: usize,
    eps: &Tensor,
    output: &DeviceTensor,
    sum_out: Option<&DeviceTensor>,
) -> TractResult<()> {
    ensure!(residual.is_some() == sum_out.is_some());
    ensure!(output.shape() == input.shape());
    ensure!(!round_scale_f16 || scale.is_some(), "round_scale_f16 requires a scale");
    let k = input.shape()[axis];
    if let Some(scale) = scale {
        ensure!(scale.datum_type() == DatumType::F32 && scale.len() == k);
    }
    if let (Some(residual), Some(sum_out)) = (residual, sum_out) {
        ensure!(residual.shape() == input.shape() && residual.datum_type() == input.datum_type());
        ensure!(sum_out.shape() == input.shape() && sum_out.datum_type() == input.datum_type());
    }
    let eps = eps.cast_to_scalar::<f32>()?;
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        if let Some(residual) = residual {
            q.retain_tensor(residual);
        }
        if let Some(sum_out) = sum_out {
            q.retain_tensor(sum_out);
        }
        let in_dt = ShaderDtype::from_datum(input.datum_type())?;
        let out_dt = ShaderDtype::from_datum(output.datum_type())?;
        let outer: usize = input.shape()[..axis].iter().product();
        let inner: usize = input.shape()[axis + 1..].iter().product();
        let rows = outer * inner;
        if rows == 0 || k == 0 {
            return Ok(());
        }
        let mut buffers = vec![get_wgpu_buffer(input)];
        let off_res = if let Some(residual) = residual {
            buffers.push(get_wgpu_buffer(residual));
            element_offset(residual, 0)
        } else {
            0
        };
        let mut off_scale = 0;
        if let Some(scale) = scale {
            q.retain_tensor(scale);
            buffers.push(get_wgpu_buffer(scale));
            off_scale = element_offset(scale, 0);
        }
        buffers.push(get_wgpu_buffer(output));
        let off_sum = if let Some(sum_out) = sum_out {
            buffers.push(get_wgpu_buffer(sum_out));
            element_offset(sum_out, 0)
        } else {
            0
        };
        let with_residual = residual.is_some();
        let scaled = scale.is_some();
        let layout = match (with_residual, scaled) {
            (false, false) => LayoutKind::Unary,
            (false, true) => LayoutKind::Chain(3),
            (true, false) => LayoutKind::TailWrites { n: 4, writes: 2 },
            (true, true) => LayoutKind::TailWrites { n: 5, writes: 2 },
        };
        let pipeline = q.context().chain_pipeline(
            ("rms_norm", in_dt, out_dt, scaled, round_scale_f16, with_residual),
            layout,
            EntryPoint::plain("rms_norm"),
            || {
                if with_residual {
                    rms_norm_residual_module(in_dt, out_dt, scaled, round_scale_f16)
                } else {
                    rms_norm_module(in_dt, out_dt, scaled, round_scale_f16)
                }
            },
        )?;
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let params = if with_residual {
            pack_u32s(&[
                element_offset(input, 0) as u32,
                off_res as u32,
                element_offset(output, 0) as u32,
                off_sum as u32,
                off_scale as u32,
                k as u32,
                inner as u32,
                rows as u32,
                eps.to_bits(),
            ])
        } else {
            pack_u32s(&[
                element_offset(input, 0) as u32,
                element_offset(output, 0) as u32,
                off_scale as u32,
                k as u32,
                inner as u32,
                rows as u32,
                eps.to_bits(),
            ])
        };
        let dyn_off = q.alloc_uniform(&params)?;
        let x = rows.min(GRID_LIMIT as usize) as u32;
        let y = rows.div_ceil(GRID_LIMIT as usize) as u32;
        q.dispatch_grid("rms_norm", &pipeline, &bg, dyn_off, [x, y, 1])
    })
}

register_wgpu_op!(tract_transformers::ops::rms_norm::RmsNorm, |source, node, op| {
    rule_if!(crate::utils::is_supported_fact(source.node_input_facts(node.id)?[0]));
    Ok(Some(Box::new(tract_gpu::ops::rms_norm::GpuRmsNorm::new(
        op.axis,
        op.eps.clone(),
        false,
        false,
        None,
        false,
        "Wgpu",
        wgpu_rms_norm_dispatch,
    ))))
});

register_wgpu_op!(tract_core::ops::nn::ScaledRmsNorm, |source, node, op| {
    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(crate::utils::is_supported_fact(input_facts[0]));
    rule_if!(input_facts[1].datum_type == DatumType::F32);
    rule_if!(op.out_dt.is_none_or(crate::utils::is_supported_dt));
    rule_if!(op.scale_dt.is_none_or(|dt| matches!(dt, DatumType::F16 | DatumType::F32)));
    Ok(Some(Box::new(tract_gpu::ops::rms_norm::GpuRmsNorm::new(
        op.axis,
        op.eps.clone(),
        true,
        false,
        op.out_dt,
        op.scale_dt == Some(DatumType::F16),
        "Wgpu",
        wgpu_rms_norm_dispatch,
    ))))
});

/// RMS norm over the middle axis of an `(outer, k, inner)` view, one workgroup
/// per normalized row, reading `input` and writing `output` dtypes. `scaled`
/// multiplies by an F32 scale along that axis; `round_f16` first rounds the
/// normalized value to f16, as a graph materializing it in f16 would.
pub fn rms_norm_module(
    input: ShaderDtype,
    output: ShaderDtype,
    scaled: bool,
    round_f16: bool,
) -> String {
    let (ti, to) = (input.wgsl(), output.wgsl());
    let bindings = if scaled {
        format!(
            "@group(0) @binding(0) var<storage, read> inp: array<{ti}>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<{to}>;
@group(0) @binding(3) var<uniform> params: Params;"
        )
    } else {
        format!(
            "@group(0) @binding(0) var<storage, read> inp: array<{ti}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{to}>;
@group(0) @binding(2) var<uniform> params: Params;"
        )
    };
    let round =
        if round_f16 { "y = unpack2x16float(pack2x16float(vec2<f32>(y, 0.0))).x;" } else { "" };
    let apply = if scaled { "y = y * scale[params.off_scale + i];" } else { "" };
    let enable =
        if input == ShaderDtype::F16 || output == ShaderDtype::F16 { "enable f16;\n" } else { "" };
    format!(
        r#"{enable}
struct Params {{
    off_in: u32,
    off_out: u32,
    off_scale: u32,
    k: u32,
    inner: u32,
    rows: u32,
    eps: f32,
}}

{bindings}

var<workgroup> partial: array<f32, {LAYER_NORM_WG}>;

@compute @workgroup_size({LAYER_NORM_WG})
fn rms_norm(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.rows) {{ return; }}
    let base = (row / params.inner) * params.k * params.inner + row % params.inner;
    var sq = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        let x = f32(inp[params.off_in + base + i * params.inner]);
        sq += x * x;
    }}
    partial[lid.x] = sq;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] += partial[lid.x + stride];
        }}
    }}
    workgroupBarrier();
    let inv = inverseSqrt(partial[0] / f32(params.k) + params.eps);
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        var y = f32(inp[params.off_in + base + i * params.inner]) * inv;
        {round}
        {apply}
        outp[params.off_out + base + i * params.inner] = {to}(y);
    }}
}}
"#
    )
}

/// RMS norm of `input + residual` over the middle axis of an `(outer, k, inner)`
/// view. The sum is computed in the input dtype, stored for the residual
/// stream, and then normalized. `scaled` multiplies by an F32 scale along that
/// axis; `round_f16` first rounds the normalized value to f16.
pub fn rms_norm_residual_module(
    input: ShaderDtype,
    output: ShaderDtype,
    scaled: bool,
    round_f16: bool,
) -> String {
    let (ti, to) = (input.wgsl(), output.wgsl());
    let bindings = if scaled {
        format!(
            "@group(0) @binding(0) var<storage, read> inp: array<{ti}>;
@group(0) @binding(1) var<storage, read> residual: array<{ti}>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<{to}>;
@group(0) @binding(4) var<storage, read_write> sum_out: array<{ti}>;
@group(0) @binding(5) var<uniform> params: Params;"
        )
    } else {
        format!(
            "@group(0) @binding(0) var<storage, read> inp: array<{ti}>;
@group(0) @binding(1) var<storage, read> residual: array<{ti}>;
@group(0) @binding(2) var<storage, read_write> outp: array<{to}>;
@group(0) @binding(3) var<storage, read_write> sum_out: array<{ti}>;
@group(0) @binding(4) var<uniform> params: Params;"
        )
    };
    let round =
        if round_f16 { "y = unpack2x16float(pack2x16float(vec2<f32>(y, 0.0))).x;" } else { "" };
    let apply = if scaled { "y = y * scale[params.off_scale + i];" } else { "" };
    let enable =
        if input == ShaderDtype::F16 || output == ShaderDtype::F16 { "enable f16;\n" } else { "" };
    format!(
        r#"{enable}
struct Params {{
    off_in: u32,
    off_res: u32,
    off_out: u32,
    off_sum: u32,
    off_scale: u32,
    k: u32,
    inner: u32,
    rows: u32,
    eps: f32,
}}

{bindings}

var<workgroup> partial: array<f32, {LAYER_NORM_WG}>;

fn summed(base: u32, i: u32) -> {ti} {{
    let at = base + i * params.inner;
    return inp[params.off_in + at] + residual[params.off_res + at];
}}

@compute @workgroup_size({LAYER_NORM_WG})
fn rms_norm(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.rows) {{ return; }}
    let base = (row / params.inner) * params.k * params.inner + row % params.inner;
    var sq = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        let v = summed(base, i);
        sum_out[params.off_sum + base + i * params.inner] = v;
        let x = f32(v);
        sq += x * x;
    }}
    partial[lid.x] = sq;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] += partial[lid.x + stride];
        }}
    }}
    workgroupBarrier();
    let inv = inverseSqrt(partial[0] / f32(params.k) + params.eps);
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        var y = f32(summed(base, i)) * inv;
        {round}
        {apply}
        outp[params.off_out + base + i * params.inner] = {to}(y);
    }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;
    use tract_core::ops::math::mul;

    use crate::test_utils::{fill, gpu_vs_cpu, transformed};

    #[test]
    fn rms_norm_runs_on_device() -> TractResult<()> {
        use tract_transformers::ops::rms_norm::RmsNorm;
        for (axis, scaled) in [(2, false), (2, true), (1, false), (1, true)] {
            let shape = [2, 77, 48];
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact(shape))?;
            let op = RmsNorm { axis, eps: rctensor0(1e-5f32) };
            let mut y = model.wire_node("rms_norm", op, &[x])?[0];
            if scaled {
                let mut gamma_shape = [1, 1, 1];
                gamma_shape[axis] = shape[axis];
                let gamma = Tensor::from_shape(&gamma_shape, &fill(3, shape[axis]))?;
                let gamma = model.add_const("gamma", gamma)?;
                y = model.wire_node("scale", mul(), &[y, gamma])?[0];
            }
            model.select_output_outlets(&[y])?;
            let gpu = transformed(model.clone())?;
            crate::ensure_wgpu_coverage(&gpu)?;
            let norms = gpu
                .nodes()
                .iter()
                .filter_map(|n| n.op_as::<tract_gpu::ops::rms_norm::GpuRmsNorm>())
                .collect::<Vec<_>>();
            ensure!(norms.len() == 1, "expected one device rms norm, got {}", norms.len());
            ensure!(norms[0].has_scale == scaled, "gamma should fold into the norm");
            let len = shape.iter().product();
            gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(5, len))?)
                .with_context(|| format!("axis {axis} scaled {scaled}"))?;
        }
        Ok(())
    }
}
