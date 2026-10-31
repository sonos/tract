use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Leaky ReLU for a contiguous f16 or f32 tensor. `alpha` scales the negative
/// values. The input is read from its byte offset as a flat run of `len`
/// elements, which is the layout a translated element-wise op has.
pub fn wgpu_leaky_relu_dispatch(
    alpha: f32,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            ("leaky_relu", dt),
            layout,
            EntryPoint::typed("leaky_relu", dt),
            || leaky_relu_module(dt),
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
            alpha.to_bits(),
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("leaky_relu", &pipeline, &bg, dyn_off, input.len() as u64)
    })
}

// LeakyRelu is an ElementWiseMiniOp, so this shares ElementWiseOp's TypeId
// with the parameter-free element-wise translator. That one declines it.
register_wgpu_op!(tract_core::ops::element_wise::ElementWiseOp, |_source, _node, op| {
    rule_if_some!(leaky = op.0.downcast_ref::<tract_core::ops::nn::LeakyRelu>());
    Ok(Some(Box::new(tract_gpu::ops::leaky_relu::GpuLeakyRelu::new(
        leaky.alpha,
        "Wgpu",
        wgpu_leaky_relu_dispatch,
    ))))
});

/// `x` when `x >= 0`, otherwise `alpha * x`. `alpha` is an f32 uniform, cast to the
/// storage type so an f16 run uses the same slope as the f16 CPU kernel.
pub fn leaky_relu_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    len: u32,
    alpha: f32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn leaky_relu_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let x = inp[params.off_in + i];
    outp[params.off_out + i] = select(x, x * {t}(params.alpha), x < {t}(0.0));
}}
"#
    ));
    s
}
