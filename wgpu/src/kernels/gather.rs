use tract_core::internal::*;
use tract_core::ops::array::Gather;
use tract_gpu::ops::gather::GpuGather;
use tract_gpu::sync::{DeviceSyncKind, sync_inputs_if_required};
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::transform::Mapping;
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn wgpu_gather_dispatch(
    data: &DeviceTensor,
    indices: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(data.rank() > axis);
    ensure!(indices.datum_type() == i64::datum_type());
    ensure!(output.datum_type() == data.datum_type());
    let axis_len = data.shape()[axis];
    let outer: usize = data.shape()[..axis].iter().product();
    let inner: usize = data.shape()[axis + 1..].iter().product();
    let n_idx = indices.len();
    ensure!(output.len() == outer * n_idx * inner);
    ensure!(axis_len <= i32::MAX as usize, "gather axis too long for 32-bit indices");
    if output.len() == 0 {
        return Ok(());
    }
    ensure!(axis_len > 0, "gather from an empty axis");
    with_wgpu_queue(|q| {
        for t in [data, indices, output] {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(data.datum_type())?;
        let layout = LayoutKind::Chain(3);
        let pipeline = q.context().chain_pipeline(
            ("gather", dt),
            layout,
            EntryPoint::typed("gather", dt),
            || gather_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(data), get_wgpu_buffer(indices), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(data, 0) as u32,
            element_offset(indices, 0) as u32,
            element_offset(output, 0) as u32,
            output.len() as u32,
            axis_len as u32,
            n_idx as u32,
            inner as u32,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("gather", &pipeline, &bg, dyn_off, output.len() as u64)
    })
}

/// A `Gather` of f16/f32 data by i64 indices. It sits outside the per-op
/// registry because the backend's dtype gate keeps every i64 input on the
/// CPU; this kernel reads the indices' words without any 64-bit arithmetic.
pub fn wgpu_gather(source: &TypedModel, node: &TypedNode) -> TractResult<Option<GpuGather>> {
    let Some(op) = node.op_as::<Gather>() else { return Ok(None) };
    let facts = source.node_input_facts(node.id)?;
    rule_if!(crate::utils::is_supported_fact(facts[0]));
    rule_if!(facts[1].datum_type == i64::datum_type() && facts[1].is_plain());
    rule_if!(op.output_type.is_none_or(|dt| dt == facts[0].datum_type));
    Ok(Some(GpuGather::new(op.axis, "Wgpu", wgpu_gather_dispatch)))
}

/// Translates the `Gather` [`wgpu_gather`] takes, both of its inputs moved to the device.
pub fn translate_gather(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    rule_if_some!(op = wgpu_gather(source, node)?);
    let device_inputs = sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
    target.wire_node(node.name.clone(), op, &device_inputs).map(Some)
}

/// `out[o, i, r] = data[o, indices[i], r]` over `(outer, axis, inner)` views,
/// one thread per output element. The i64 indices are read as pairs of u32
/// words, low word first: an index must fit in 32 bits, and a negative one
/// counts from the end of the axis.
pub fn gather_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_data: u32,
    off_idx: u32,
    off_out: u32,
    len: u32,
    axis_len: u32,
    n_idx: u32,
    inner: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> data: array<{t}>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn gather_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let r = i % params.inner;
    let n = (i / params.inner) % params.n_idx;
    let o = i / (params.inner * params.n_idx);
    var ix = bitcast<i32>(indices[2u * (params.off_idx + n)]);
    if (ix < 0) {{ ix += i32(params.axis_len); }}
    let at = u32(clamp(ix, 0, i32(params.axis_len) - 1));
    outp[params.off_out + i] = data[params.off_data + (o * params.axis_len + at) * params.inner + r];
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;

    use crate::test_utils::{fill, gpu_vs_cpu, transformed};

    /// Token embedding lookups and a gather along a middle axis, by i64 indices
    /// fed from the host, negative ones included.
    #[test]
    fn gather_runs_on_device() -> TractResult<()> {
        use tract_core::ops::array::Gather;
        use tract_gpu::ops::gather::GpuGather;
        for (data_shape, axis, f16_) in
            [([1000usize, 1, 64], 0, false), ([1000, 1, 64], 0, true), ([3, 50, 6], 1, false)]
        {
            let dt = if f16_ { f16::datum_type() } else { f32::datum_type() };
            let len = data_shape.iter().product();
            let data = Tensor::from_shape(&data_shape, &fill(7, len))?.cast_to_dt(dt)?.into_owned();
            let mut model = TypedModel::default();
            let indices = model.add_source("indices", i64::fact([2, 5]))?;
            let data = model.add_const("table", data)?;
            let y = model.wire_node("gather", Gather::new(axis), &[data, indices])?[0];
            model.select_output_outlets(&[y])?;
            let gpu = transformed(model.clone())?;
            crate::ensure_wgpu_coverage(&gpu)?;
            ensure!(gpu.nodes().iter().any(|n| n.op_is::<GpuGather>()), "gather stayed on the CPU");
            let n = data_shape[axis] as i64;
            let ix = tensor2(&[[0i64, 3, n - 1, -1, -n], [17 % n, 2, 1, -2, 5]]);
            gpu_vs_cpu(model, ix)
                .with_context(|| format!("{data_shape:?} axis {axis} f16 {f16_}"))?;
        }
        Ok(())
    }
}
