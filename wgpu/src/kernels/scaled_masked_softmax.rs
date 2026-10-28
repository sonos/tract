use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, pack_u32s, scaled_masked_softmax_module,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

const MAX_RANK: usize = 5;

fn left_padded<T: Copy>(dims: &[T], fill: T) -> [T; MAX_RANK] {
    let mut padded = [fill; MAX_RANK];
    padded[MAX_RANK - dims.len()..].copy_from_slice(dims);
    padded
}

/// The strides `t` is read through over `shape`, zero along the axes it
/// broadcasts.
fn broadcast_strides(t: &DeviceTensor, shape: &[usize; MAX_RANK]) -> TractResult<[u32; MAX_RANK]> {
    let t_shape = left_padded(t.shape(), 1);
    let t_strides = left_padded(t.strides(), 0);
    let mut strides = [0; MAX_RANK];
    for axis in 0..MAX_RANK {
        if t_shape[axis] == shape[axis] {
            ensure!(t_strides[axis] >= 0, "negative stride in {:?}", t.strides());
            strides[axis] = t_strides[axis] as u32;
        } else {
            ensure!(t_shape[axis] == 1, "{:?} does not broadcast to {shape:?}", t.shape());
        }
    }
    Ok(strides)
}

/// Dispatch for `GpuScaledMaskedSoftmax` with a float mask of the input's
/// dtype. A bool mask never reaches it: the backend keeps bool tensors on the
/// CPU.
pub fn wgpu_scaled_masked_softmax_dispatch(
    input: &DeviceTensor,
    scale: &Tensor,
    mask: &DeviceTensor,
    post_softmax_mask: bool,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    ensure!(mask.datum_type() == input.datum_type(), "wgpu takes float masks only");
    ensure!(!post_softmax_mask, "post_softmax_mask needs a bool mask");
    ensure!((2..=MAX_RANK).contains(&input.rank()) && mask.rank() == input.rank());
    let shape = left_padded(input.shape(), 1);
    let x_strides = broadcast_strides(input, &shape)?;
    let mask_strides = broadcast_strides(mask, &shape)?;
    let k = shape[MAX_RANK - 1];
    let rows = input.len() / k.max(1);
    if rows == 0 || k == 0 {
        return Ok(());
    }
    let scale = scale.cast_to_scalar::<f32>()?;
    with_wgpu_queue(|q| {
        for t in [input, mask, output] {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(3);
        let pipeline = q.context().chain_pipeline(
            ("scaled_masked_softmax", dt),
            layout,
            EntryPoint::typed("scaled_masked_softmax", dt),
            || scaled_masked_softmax_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(mask), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let mut params = vec![
            element_offset(input, 0) as u32,
            element_offset(mask, 0) as u32,
            element_offset(output, 0) as u32,
            shape[1] as u32,
            shape[2] as u32,
            shape[3] as u32,
            k as u32,
            rows as u32,
            scale.to_bits(),
            x_strides[4],
            mask_strides[4],
            0,
        ];
        params.extend(&x_strides[..4]);
        params.extend(&mask_strides[..4]);
        let dyn_off = q.alloc_uniform(&pack_u32s(&params))?;
        let x = rows.min(GRID_LIMIT as usize) as u32;
        let y = rows.div_ceil(GRID_LIMIT as usize) as u32;
        q.dispatch_grid("scaled_masked_softmax", &pipeline, &bg, dyn_off, [x, y, 1])
    })
}

register_wgpu_op!(
    tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax,
    |source, node, op| {
        let facts = source.node_input_facts(node.id)?;
        rule_if!(!op.post_softmax_mask);
        rule_if!(facts[1].datum_type == facts[0].datum_type);
        rule_if!((2..=MAX_RANK).contains(&facts[0].rank()) && facts[1].rank() == facts[0].rank());
        Ok(Some(Box::new(tract_gpu::ops::scaled_masked_softmax::GpuScaledMaskedSoftmax::new(
            op.scale.clone(),
            false,
            "Wgpu",
            wgpu_scaled_masked_softmax_dispatch,
        ))))
    }
);
