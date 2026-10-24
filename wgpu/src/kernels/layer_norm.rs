use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, layer_norm_module, pack_u32s,
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
