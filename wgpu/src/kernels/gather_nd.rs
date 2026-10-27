use tract_core::internal::*;
use tract_core::ops::array::GatherNd;
use tract_gpu::ops::gather_nd::{GpuGatherNd, MAX_COORDS};
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{EntryPoint, LayoutKind, ShaderDtype, gather_nd_module, pack_u32s};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Gathers the slices of `data` that rows of f32 `indices` point at, after `batch_dims` axes
/// both share.
pub fn wgpu_gather_nd_dispatch(
    batch_dims: usize,
    data: &DeviceTensor,
    indices: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        for t in [data, indices, output] {
            q.retain_tensor(t);
        }
        ensure!(indices.datum_type() == f32::datum_type());
        ensure!(output.datum_type() == data.datum_type());
        let n = *indices.shape().last().context("GatherNd indices of rank 0")?;
        ensure!(0 < n && n <= MAX_COORDS && batch_dims + n <= data.rank());
        let shape = data.shape();
        let batch_stride: usize = shape[batch_dims..].iter().product();
        let inner: usize = shape[batch_dims + n..].iter().product();
        let mut dims = [0u32; MAX_COORDS];
        let mut strides = [0u32; MAX_COORDS];
        for k in 0..n {
            dims[k] = shape[batch_dims + k] as u32;
            strides[k] = shape[batch_dims + k + 1..].iter().product::<usize>() as u32;
        }
        let batch: usize = shape[..batch_dims].iter().product();
        let rows = indices.len() / n / batch.max(1);
        let dt = ShaderDtype::from_datum(data.datum_type())?;
        let layout = LayoutKind::Chain(3);
        let pipeline = q.context().chain_pipeline(
            ("gather_nd", dt),
            layout,
            EntryPoint::typed("gather_nd", dt),
            || gather_nd_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(data), get_wgpu_buffer(indices), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let mut params = vec![
            element_offset(data, 0) as u32,
            element_offset(indices, 0) as u32,
            element_offset(output, 0) as u32,
            output.len() as u32,
            inner as u32,
            rows as u32,
            n as u32,
            batch_stride as u32,
        ];
        params.extend_from_slice(&dims);
        params.extend_from_slice(&strides);
        let dyn_off = q.alloc_uniform(&pack_u32s(&params))?;
        q.dispatch("gather_nd", &pipeline, &bg, dyn_off, output.len() as u64)
    })
}

register_wgpu_op!(GatherNd, |source, node, op| {
    rule_if!(GpuGatherNd::is_supported(&source.node_input_facts(node.id)?));
    Ok(Some(Box::new(GpuGatherNd::new(op.batch_dims, "Wgpu", wgpu_gather_nd_dispatch))))
});
