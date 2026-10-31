use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn wgpu_nan_replace_dispatch(
    value: f32,
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
            ("nan_replace", dt),
            layout,
            EntryPoint::typed("nan_replace", dt),
            || nan_replace_module(dt),
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
            value.to_bits(),
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("nan_replace", &pipeline, &bg, dyn_off, input.len() as u64)
    })
}

/// Replaces each NaN of the input with `value`. NaN is told by its bits, on the value widened
/// to f32 (which keeps a NaN a NaN): WGSL lets a compiler assume no NaN and fold `x != x`.
pub fn nan_replace_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    len: u32,
    value: f32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn nan_replace_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let x = inp[params.off_in + i];
    let bits = bitcast<u32>(f32(x));
    outp[params.off_out + i] = select(x, {t}(params.value), (bits & 0x7fffffffu) > 0x7f800000u);
}}
"#
    ));
    s
}
