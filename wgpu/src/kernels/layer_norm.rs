use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LAYER_NORM_WG, LayoutKind, ShaderDtype, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

pub fn wgpu_layer_norm_dispatch(
    input: &DeviceTensor,
    affine: Option<(&DeviceTensor, &DeviceTensor)>,
    axis: usize,
    eps: f32,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    let k = input.shape()[axis];
    if let Some((scale, bias)) = affine {
        for t in [scale, bias] {
            ensure!(t.datum_type() == DatumType::F32 && t.len() == k);
        }
    }
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let outer: usize = input.shape()[..axis].iter().product();
        let inner: usize = input.shape()[axis + 1..].iter().product();
        let rows = outer * inner;
        if rows == 0 || k == 0 {
            return Ok(());
        }
        let mut buffers = vec![get_wgpu_buffer(input)];
        let (mut off_scale, mut off_bias) = (0, 0);
        if let Some((scale, bias)) = affine {
            q.retain_tensor(scale);
            q.retain_tensor(bias);
            buffers.push(get_wgpu_buffer(scale));
            buffers.push(get_wgpu_buffer(bias));
            off_scale = element_offset(scale, 0);
            off_bias = element_offset(bias, 0);
        }
        buffers.push(get_wgpu_buffer(output));
        let layout = if affine.is_some() { LayoutKind::Chain(4) } else { LayoutKind::Unary };
        let pipeline = q.context().chain_pipeline(
            ("layer_norm", dt, affine.is_some()),
            layout,
            EntryPoint::typed("layer_norm", dt),
            || layer_norm_module(dt, affine.is_some()),
        )?;
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            off_scale as u32,
            off_bias as u32,
            k as u32,
            inner as u32,
            rows as u32,
            eps.to_bits(),
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        let x = rows.min(GRID_LIMIT as usize) as u32;
        let y = rows.div_ceil(GRID_LIMIT as usize) as u32;
        q.dispatch_grid("layer_norm", &pipeline, &bg, dyn_off, [x, y, 1])
    })
}

register_wgpu_op!(tract_core::ops::nn::LayerNorm, |source, node, op| {
    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(crate::utils::is_supported_fact(input_facts[0]));
    rule_if!(input_facts[1..].iter().all(|f| f.datum_type == DatumType::F32));
    Ok(Some(Box::new(tract_gpu::ops::layer_norm::GpuLayerNorm {
        axis: op.axis,
        eps: op.eps.clone(),
        backend_name: "Wgpu",
        dispatch: wgpu_layer_norm_dispatch,
    })))
});

/// Layer norm over the middle axis of an `(outer, k, inner)` view, one
/// workgroup per normalized row. `affine` multiplies by an F32 scale and adds
/// an F32 bias along that axis.
pub fn layer_norm_module(dt: ShaderDtype, affine: bool) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let (bindings, apply) = if affine {
        (
            format!(
                "@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(4) var<uniform> params: Params;"
            ),
            "y = y * scale[params.off_scale + i] + bias[params.off_bias + i];",
        )
    } else {
        (
            format!(
                "@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;"
            ),
            "",
        )
    };
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    off_scale: u32,
    off_bias: u32,
    k: u32,
    inner: u32,
    rows: u32,
    eps: f32,
}}

{bindings}

var<workgroup> partial: array<f32, {LAYER_NORM_WG}>;

fn workgroup_sum(lid: u32, v: f32) -> f32 {{
    partial[lid] = v;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid < stride) {{
            partial[lid] += partial[lid + stride];
        }}
    }}
    workgroupBarrier();
    let total = partial[0];
    workgroupBarrier();
    return total;
}}

@compute @workgroup_size({LAYER_NORM_WG})
fn layer_norm_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.rows) {{ return; }}
    let base = (row / params.inner) * params.k * params.inner + row % params.inner;
    var sum = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        sum += f32(inp[params.off_in + base + i * params.inner]);
    }}
    let mean = workgroup_sum(lid.x, sum) / f32(params.k);
    var sq = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        let d = f32(inp[params.off_in + base + i * params.inner]) - mean;
        sq += d * d;
    }}
    let inv = inverseSqrt(workgroup_sum(lid.x, sq) / f32(params.k) + params.eps);
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        var y = (f32(inp[params.off_in + base + i * params.inner]) - mean) * inv;
        {apply}
        outp[params.off_out + base + i * params.inner] = {t}(y);
    }}
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;

    use crate::test_utils::{fill, transformed};

    #[test]
    fn layer_norm_runs_on_device() -> TractResult<()> {
        use tract_core::ops::math::{add, mul};
        use tract_core::ops::nn::LayerNorm;
        for (axis, affine) in [(2, false), (2, true), (1, false)] {
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact([2, 77, 48]))?;
            let op = LayerNorm { axis, eps: rctensor0(1e-5f32) };
            let mut y = model.wire_node("layer_norm", op, &[x])?[0];
            if affine {
                let gamma =
                    model.add_const("gamma", Tensor::from_shape(&[1, 1, 48], &fill(1, 48))?)?;
                let beta =
                    model.add_const("beta", Tensor::from_shape(&[1, 1, 48], &fill(2, 48))?)?;
                y = model.wire_node("scale", mul(), &[y, gamma])?[0];
                y = model.wire_node("shift", add(), &[y, beta])?[0];
            }
            model.select_output_outlets(&[y])?;
            let model = transformed(model)?;
            crate::ensure_wgpu_coverage(&model)?;
            let norms = model
                .nodes()
                .iter()
                .filter(|n| n.op_is::<tract_gpu::ops::layer_norm::GpuLayerNorm>())
                .collect::<Vec<_>>();
            ensure!(norms.len() == 1, "expected one device layer norm, got {}", norms.len());
            let inputs = if affine { 3 } else { 1 };
            ensure!(norms[0].inputs.len() == inputs, "gamma and beta should fold into the norm");
        }
        Ok(())
    }
}
