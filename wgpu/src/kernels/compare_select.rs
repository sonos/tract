use tract_core::internal::*;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::logic::Iff;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::compare_select::{Comparison, GpuCompareSelect};
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    AT8, EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, broadcast_strides, pack_u32s,
    preamble,
};
use crate::transform::Mapping;
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn wgpu_compare_select_dispatch(
    comparison: Comparison,
    a: &DeviceTensor,
    b: &DeviceTensor,
    t: &DeviceTensor,
    f: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let inputs = [a, b, t, f];
    let rank = output.rank();
    ensure!(rank <= 8 && inputs.iter().all(|i| i.rank() == rank));
    ensure!(inputs.iter().all(|i| i.datum_type() == output.datum_type()));
    with_wgpu_queue(|q| {
        for i in inputs {
            q.retain_tensor(i);
        }
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(output.datum_type())?;
        let layout = LayoutKind::Chain(5);
        let name = match comparison {
            Comparison::Eq => "compare_select_eq",
            Comparison::Ne => "compare_select_ne",
            Comparison::Lt => "compare_select_lt",
            Comparison::Lte => "compare_select_lte",
            Comparison::Gt => "compare_select_gt",
            Comparison::Gte => "compare_select_gte",
        };
        let pipeline =
            q.context().chain_pipeline((name, dt), layout, EntryPoint::typed(name, dt), || {
                compare_select_module(dt)
            })?;
        let buffers: Vec<_> =
            inputs.iter().map(|i| get_wgpu_buffer(i)).chain([get_wgpu_buffer(output)]).collect();
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let mut vals: Vec<u32> = inputs.iter().map(|i| element_offset(i, 0) as u32).collect();
        vals.extend([element_offset(output, 0) as u32, rank as u32, output.len() as u32, 0]);
        for i in inputs {
            vals.extend(broadcast_strides(i.shape(), i.strides(), output.shape()));
        }
        let mut shape = [1u32; 8];
        for (ix, d) in output.shape().iter().enumerate() {
            shape[ix] = *d as u32;
        }
        vals.extend(shape);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        q.dispatch("compare_select", &pipeline, &bg, dyn_off, output.len() as u64)
    })
}

/// Translates `Iff(Comp(a, b), t, f)` into one [`GpuCompareSelect`] of `a, b, t, f`, when the
/// four are float tensors of one type and rank: wgpu has no boolean tensor to carry the
/// condition. The comparison node is left to the CPU, where nothing reads it once every
/// `Iff` it fed is translated.
pub fn translate_compare_select(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    rule_if!(node.op_is::<Iff>());
    let cond = source.node(node.inputs[0].node);
    rule_if_some!(bin = cond.op_as::<TypedBinOp>());
    rule_if_some!(comparison = Comparison::from_mini_op(&*bin.0));
    let outlets = [cond.inputs[0], cond.inputs[1], node.inputs[1], node.inputs[2]];
    let facts: TVec<&TypedFact> =
        outlets.iter().map(|o| source.outlet_fact(*o)).collect::<TractResult<_>>()?;
    let dt = facts[2].datum_type;
    rule_if!(facts.iter().all(|f| crate::utils::is_supported_fact(f) && f.datum_type == dt));
    rule_if!(facts.iter().all(|f| f.rank() == facts[0].rank()) && facts[0].rank() <= 8);
    let mut inputs = tvec![];
    for (ix, outlet) in outlets.iter().enumerate() {
        let input = mapping[outlet];
        let input = if target.outlet_fact(input)?.as_device_fact().is_some() {
            input
        } else {
            target.wire_node(
                format!("{}.to-device-{ix}", node.name),
                tract_gpu::sync::DeviceSync::new(tract_gpu::sync::DeviceSyncKind::ToDevice),
                &[input],
            )?[0]
        };
        inputs.push(input);
    }
    let op = GpuCompareSelect::new(comparison, "Wgpu", wgpu_compare_select_dispatch);
    target.wire_node(node.name.clone(), op, &inputs).map(Some)
}

/// `cmp(a, b) ? t : f` over four broadcasting inputs, one entry point per comparison.
pub fn compare_select_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_a: u32,
    off_b: u32,
    off_t: u32,
    off_f: u32,
    off_out: u32,
    rank: u32,
    len: u32,
    _p0: u32,
    a_s0: vec4<u32>,
    a_s1: vec4<u32>,
    b_s0: vec4<u32>,
    b_s1: vec4<u32>,
    t_s0: vec4<u32>,
    t_s1: vec4<u32>,
    f_s0: vec4<u32>,
    f_s1: vec4<u32>,
    sh0: vec4<u32>,
    sh1: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> in_a: array<{t}>;
@group(0) @binding(1) var<storage, read> in_b: array<{t}>;
@group(0) @binding(2) var<storage, read> in_t: array<{t}>;
@group(0) @binding(3) var<storage, read> in_f: array<{t}>;
@group(0) @binding(4) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(5) var<uniform> params: Params;

{AT8}

struct Picked {{
    a: {t},
    b: {t},
    t: {t},
    f: {t},
}}

fn pick(i: u32) -> Picked {{
    var rest = i;
    var ia = params.off_a;
    var ib = params.off_b;
    var it = params.off_t;
    var jf = params.off_f;
    for (var k = 0u; k < params.rank; k++) {{
        let axis = params.rank - 1u - k;
        let dim = max(at8(params.sh0, params.sh1, axis), 1u);
        let c = rest % dim;
        rest = rest / dim;
        ia += c * at8(params.a_s0, params.a_s1, axis);
        ib += c * at8(params.b_s0, params.b_s1, axis);
        it += c * at8(params.t_s0, params.t_s1, axis);
        jf += c * at8(params.f_s0, params.f_s1, axis);
    }}
    return Picked(in_a[ia], in_b[ib], in_t[it], in_f[jf]);
}}
"#
    ));
    for (name, op) in
        [("eq", "=="), ("ne", "!="), ("lt", "<"), ("lte", "<="), ("gt", ">"), ("gte", ">=")]
    {
        s.push_str(&format!(
            r#"
@compute @workgroup_size({WORKGROUP})
fn compare_select_{name}_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let p = pick(i);
    outp[params.off_out + i] = select(p.f, p.t, p.a {op} p.b);
}}
"#
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;
    use tract_core::ops::binary::TypedBinOp;
    use tract_core::ops::logic::{Iff, comp_eq, comp_gt};
    use tract_gpu::ops::compare_select::GpuCompareSelect;

    use crate::test_utils::{gpu_vs_cpu, transformed};

    /// A masked fill, `Iff(x * m == 0, -inf, x)` with the mask broadcast over rows, then a
    /// PReLU of it, `Iff(y > 0, y, alpha * y)`: two comparisons, each run on the device as one
    /// op, with no boolean tensor left on the CPU.
    #[test]
    fn compare_select_runs_on_device() -> TractResult<()> {
        for dt in [f32::datum_type(), f16::datum_type()] {
            let build = || -> TractResult<TypedModel> {
                let konst =
                    |model: &mut TypedModel, name: &str, t: Tensor| -> TractResult<OutletId> {
                        model.add_const(name, t.cast_to_dt(dt)?.into_owned())
                    };
                let mut model = TypedModel::default();
                let x = model.add_source("x", dt.fact([3, 5]))?;
                let m = konst(&mut model, "m", tensor2(&[[1f32, 1., 0., 1., 0.]]))?;
                let zero = konst(&mut model, "zero", tensor2(&[[0f32]]))?;
                let fill = konst(&mut model, "fill", tensor2(&[[f32::NEG_INFINITY]]))?;
                let alpha = konst(&mut model, "alpha", tensor2(&[[0.1f32], [0.2], [0.3]]))?;
                let mul = || TypedBinOp(Box::new(tract_core::ops::math::Mul), None);
                let masked = model.wire_node("masked", mul(), &[x, m])?[0];
                let is_masked =
                    model.wire_node("is_masked", TypedBinOp(comp_eq(), None), &[masked, zero])?[0];
                let y = model.wire_node("y", Iff, &[is_masked, fill, x])?[0];
                let scaled = model.wire_node("scaled", mul(), &[y, alpha])?[0];
                let positive =
                    model.wire_node("positive", TypedBinOp(comp_gt(), None), &[y, zero])?[0];
                let z = model.wire_node("z", Iff, &[positive, y, scaled])?[0];
                model.select_output_outlets(&[z])?;
                Ok(model)
            };
            let fused = transformed(build()?)?;
            crate::ensure_wgpu_coverage(&fused)?;
            let selects = fused.nodes().iter().filter(|n| n.op_is::<GpuCompareSelect>()).count();
            ensure!(selects == 2, "{selects} compare-selects in {fused}");
            let data: Vec<f32> = (0..15).map(|i| (i as f32 - 7.0) * 0.4).collect();
            let input = Tensor::from_shape(&[3, 5], &data)?.cast_to_dt(dt)?.into_owned();
            gpu_vs_cpu(build()?, input).with_context(|| format!("{dt:?}"))?;
        }
        Ok(())
    }
}
