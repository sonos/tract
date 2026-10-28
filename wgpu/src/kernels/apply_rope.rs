use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ShaderDtype, apply_rope_module, pack_u32s, rotate_half_module,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

const MAX_RANK: usize = 4;

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

pub fn wgpu_apply_rope_dispatch(
    input: &DeviceTensor,
    cos: &DeviceTensor,
    sin: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    ensure!(cos.datum_type() == input.datum_type() && sin.datum_type() == input.datum_type());
    ensure!(input.rank() <= MAX_RANK && cos.rank() <= input.rank() && sin.rank() <= input.rank());
    let shape = left_padded(input.shape(), 1);
    ensure!(
        shape[MAX_RANK - 1].is_multiple_of(2),
        "ApplyRope needs an even innermost axis: {shape:?}"
    );
    let x_strides = broadcast_strides(input, &shape)?;
    let cos_strides = broadcast_strides(cos, &shape)?;
    let sin_strides = broadcast_strides(sin, &shape)?;
    let half = shape[MAX_RANK - 1] / 2;
    let len = input.len() / 2;
    if len == 0 {
        return Ok(());
    }
    with_wgpu_queue(|q| {
        for t in [input, cos, sin, output] {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(4);
        let pipeline = q.context().chain_pipeline(
            ("apply_rope", dt),
            layout,
            EntryPoint::typed("apply_rope", dt),
            || apply_rope_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(input),
                get_wgpu_buffer(cos),
                get_wgpu_buffer(sin),
                get_wgpu_buffer(output),
            ],
            q.uniform(),
        )?;
        let mut params = vec![
            element_offset(input, 0) as u32,
            element_offset(cos, 0) as u32,
            element_offset(sin, 0) as u32,
            element_offset(output, 0) as u32,
            shape[1] as u32,
            shape[2] as u32,
            half as u32,
            len as u32,
        ];
        params.extend(x_strides);
        params.extend(cos_strides);
        params.extend(sin_strides);
        let dyn_off = q.alloc_uniform(&pack_u32s(&params))?;
        q.dispatch("apply_rope", &pipeline, &bg, dyn_off, len as u64)
    })
}

pub fn wgpu_rotate_half_dispatch(input: &DeviceTensor, output: &DeviceTensor) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    ensure!(input.rank() > 0);
    let inner = input.shape()[input.rank() - 1];
    ensure!(
        inner.is_multiple_of(2),
        "RotateHalf needs an even innermost axis: {:?}",
        input.shape()
    );
    let len = input.len() / 2;
    if len == 0 {
        return Ok(());
    }
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            ("rotate_half", dt),
            layout,
            EntryPoint::typed("rotate_half", dt),
            || rotate_half_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            (inner / 2) as u32,
            len as u32,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("rotate_half", &pipeline, &bg, dyn_off, len as u64)
    })
}

register_wgpu_op!(tract_transformers::ops::apply_rope::ApplyRope, |source, node, _op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts.iter().all(|f| f.datum_type == facts[0].datum_type));
    rule_if!(facts[0].rank() <= MAX_RANK && facts.iter().all(|f| f.rank() <= facts[0].rank()));
    Ok(Some(Box::new(tract_gpu::ops::apply_rope::GpuApplyRope::new(
        "Wgpu",
        wgpu_apply_rope_dispatch,
    ))))
});

register_wgpu_op!(tract_transformers::ops::apply_rope::RotateHalf, |_source, _node, _op| {
    Ok(Some(Box::new(tract_gpu::ops::rotate_half::GpuRotateHalf::new(
        "Wgpu",
        wgpu_rotate_half_dispatch,
    ))))
});
