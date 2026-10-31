use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Tanh-form GELU for a contiguous f16 or f32 tensor. `fast_impl` uses `x^2`
/// inside the tanh; the canonical form uses `x^3`. The input is read from its
/// byte offset as a flat run of `len` elements.
pub fn wgpu_gelu_approximate_dispatch(
    fast_impl: bool,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let stem = if fast_impl { "gelu_approx_fast" } else { "gelu_approx" };
        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            ("gelu_approximate", dt, fast_impl),
            layout,
            EntryPoint::typed(stem, dt),
            || gelu_approximate_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            input.len() as u32,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch(stem, &pipeline, &bg, dyn_off, input.len() as u64)
    })
}

// GeluApproximate is an ElementWiseMiniOp, so this shares ElementWiseOp's
// TypeId with the parameter-free element-wise translator. That one declines it.
register_wgpu_op!(tract_core::ops::element_wise::ElementWiseOp, |source, node, op| {
    rule_if_some!(ew = op.0.downcast_ref::<tract_core::ops::nn::GeluApproximate>());
    rule_if!(matches!(
        source.node_input_facts(node.id)?[0].datum_type,
        DatumType::F32 | DatumType::F16
    ));
    Ok(Some(Box::new(tract_gpu::ops::gelu_approximate::GpuGeluApproximate::new(
        ew.fast_impl,
        "Wgpu",
        wgpu_gelu_approximate_dispatch,
    ))))
});

/// Tanh-form GELU, `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^p)))`.
/// `p` is 3 for `gelu_approx_*` and 2 for `gelu_approx_fast_*`.
///
/// The f32 canonical path uses the same rational tanh as `Func::Gelu`. WGSL
/// `tanh` is what the f16 op and the f32 fast op call.
pub fn gelu_approximate_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let accurate = match dt {
        ShaderDtype::F32 => "true",
        ShaderDtype::F16 => "false",
    };
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    len: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEF: f32 = 0.044715;

fn stanh(x: f32) -> f32 {{
    let c = clamp(x, -7.5, 7.5);
    let x2 = c * c;
    var p = -8.488492677e-14;
    p = x2 * p + 5.277853000e-11;
    p = x2 * p - 2.022500419e-8;
    p = x2 * p + 0.00001115424833;
    p = x2 * p + 0.003103950131;
    p = x2 * p + 0.1308400453;
    p = x2 * p + 0.9999999934;
    p = p * c;
    var q = 0.0002546136580;
    q = x2 * q + 0.02449515379;
    q = x2 * q + 0.4641733162;
    q = x2 * q + 1.0;
    return p / q;
}}

fn gelu_body(x: f32, cubic: bool, accurate: bool) -> f32 {{
    let x2 = x * x;
    let xp = select(x2, x2 * x, cubic);
    let inner = SQRT_2_OVER_PI * (x + GELU_COEF * xp);
    let th = select(tanh(inner), stanh(inner), accurate);
    return 0.5 * x * (1.0 + th);
}}

@compute @workgroup_size({WORKGROUP})
fn gelu_approx_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    outp[params.off_out + i] = {t}(gelu_body(f32(inp[params.off_in + i]), true, {accurate}));
}}

@compute @workgroup_size({WORKGROUP})
fn gelu_approx_fast_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    outp[params.off_out + i] = {t}(gelu_body(f32(inp[params.off_in + i]), false, false));
}}
"#
    ));
    s
}
