use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::logic::Iff;
use tract_core::ops::math::IsNan;
use tract_gpu::ops::nan_replace::GpuNanReplace;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::transform::{Mapping, input_on_device};
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

/// Translates `Iff(IsNan(x), value, x)` into one [`GpuNanReplace`] of `x`; the condition is
/// not wired.
pub fn translate_nan_replace(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    rule_if_some!(fill = nan_replace_fill(source, node)?);
    let input = input_on_device(node, target, mapping, 2)?;
    let op = GpuNanReplace::new(fill, "Wgpu", wgpu_nan_replace_dispatch);
    target.wire_node(node.name.clone(), op, &[input]).map(Some)
}

/// The value `Iff(IsNan(x), value, x)` puts in place of each NaN of `x`, when `node` is that
/// and `x` is a tensor the kernels take: wgpu has no boolean tensor to carry the condition.
fn nan_replace_fill(source: &TypedModel, node: &TypedNode) -> TractResult<Option<f32>> {
    rule_if!(node.op_is::<Iff>());
    let cond = source.node(node.inputs[0].node);
    rule_if!(
        cond.op_as::<ElementWiseOp>().is_some_and(|ew| ew.0.is::<IsNan>())
            && cond.inputs[0] == node.inputs[2]
    );
    let x = source.outlet_fact(node.inputs[2])?;
    rule_if!(crate::utils::is_supported_fact(x));
    rule_if_some!(fill = source.outlet_fact(node.inputs[1])?.konst.as_ref());
    rule_if!(fill.len() == 1 && fill.datum_type().is_float());
    Ok(Some(fill.cast_to_scalar::<f32>()?))
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

#[cfg(test)]
mod tests {
    use tract_core::internal::*;

    use crate::test_utils::{fill, gpu_vs_cpu, transformed};

    /// `Iff(IsNan(x), value, x)` runs on the device as one op, NaNs replaced and other values
    /// untouched, with no boolean tensor left on the CPU.
    #[test]
    fn nan_replace_runs_on_device() -> TractResult<()> {
        use tract_core::ops::logic::Iff;
        use tract_core::ops::math::is_nan;
        for (dt, value) in [(f32::datum_type(), 0.0f32), (f16::datum_type(), -2.5)] {
            let build = || -> TractResult<TypedModel> {
                let mut model = TypedModel::default();
                let x = model.add_source("x", dt.fact([3, 5]))?;
                let nan = model.wire_node("nan", is_nan(), &[x])?[0];
                let value = tensor2(&[[value]]).cast_to_dt(dt)?.into_owned();
                let value = model.add_const("value", value)?;
                let y = model.wire_node("y", Iff, &[nan, value, x])?[0];
                model.select_output_outlets(&[y])?;
                Ok(model)
            };
            crate::ensure_wgpu_coverage(&transformed(build()?)?)?;
            let mut data = fill(1, 15);
            for i in [0, 6, 14] {
                data[i] = f32::NAN;
            }
            let input = Tensor::from_shape(&[3, 5], &data)?.cast_to_dt(dt)?.into_owned();
            gpu_vs_cpu(build()?, input).with_context(|| format!("{dt:?}"))?;
        }
        Ok(())
    }
}
