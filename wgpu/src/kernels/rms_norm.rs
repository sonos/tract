use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, pack_u32s, rms_norm_module,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Dispatch for `GpuRmsNorm`. The fused residual add is not implemented here:
/// it needs two writable outputs, and `fuse_rms_norm_residual` leaves wgpu
/// norms alone.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_rms_norm_dispatch(
    input: &DeviceTensor,
    residual: Option<&DeviceTensor>,
    scale: Option<&DeviceTensor>,
    round_scale_f16: bool,
    axis: usize,
    eps: &Tensor,
    output: &DeviceTensor,
    sum_out: Option<&DeviceTensor>,
) -> TractResult<()> {
    ensure!(residual.is_none() && sum_out.is_none(), "wgpu RmsNorm has no fused residual");
    ensure!(output.shape() == input.shape());
    ensure!(!round_scale_f16 || scale.is_some(), "round_scale_f16 requires a scale");
    let k = input.shape()[axis];
    if let Some(scale) = scale {
        ensure!(scale.datum_type() == DatumType::F32 && scale.len() == k);
    }
    let eps = eps.cast_to_scalar::<f32>()?;
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let in_dt = ShaderDtype::from_datum(input.datum_type())?;
        let out_dt = ShaderDtype::from_datum(output.datum_type())?;
        let outer: usize = input.shape()[..axis].iter().product();
        let inner: usize = input.shape()[axis + 1..].iter().product();
        let rows = outer * inner;
        if rows == 0 || k == 0 {
            return Ok(());
        }
        let mut buffers = vec![get_wgpu_buffer(input)];
        let mut off_scale = 0;
        if let Some(scale) = scale {
            q.retain_tensor(scale);
            buffers.push(get_wgpu_buffer(scale));
            off_scale = element_offset(scale, 0);
        }
        buffers.push(get_wgpu_buffer(output));
        let scaled = scale.is_some();
        let layout = if scaled { LayoutKind::Chain(3) } else { LayoutKind::Unary };
        let pipeline = q.context().chain_pipeline(
            ("rms_norm", in_dt, out_dt, scaled, round_scale_f16),
            layout,
            EntryPoint::plain("rms_norm"),
            || rms_norm_module(in_dt, out_dt, scaled, round_scale_f16),
        )?;
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            off_scale as u32,
            k as u32,
            inner as u32,
            rows as u32,
            eps.to_bits(),
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        let x = rows.min(GRID_LIMIT as usize) as u32;
        let y = rows.div_ceil(GRID_LIMIT as usize) as u32;
        q.dispatch_grid("rms_norm", &pipeline, &bg, dyn_off, [x, y, 1])
    })
}

register_wgpu_op!(tract_transformers::ops::rms_norm::RmsNorm, |source, node, op| {
    rule_if!(crate::utils::is_supported_fact(source.node_input_facts(node.id)?[0]));
    Ok(Some(Box::new(tract_gpu::ops::rms_norm::GpuRmsNorm::new(
        op.axis,
        op.eps.clone(),
        false,
        false,
        None,
        false,
        "Wgpu",
        wgpu_rms_norm_dispatch,
    ))))
});

register_wgpu_op!(tract_core::ops::nn::ScaledRmsNorm, |source, node, op| {
    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(crate::utils::is_supported_fact(input_facts[0]));
    rule_if!(input_facts[1].datum_type == DatumType::F32);
    rule_if!(op.out_dt.is_none_or(crate::utils::is_supported_dt));
    rule_if!(op.scale_dt.is_none_or(|dt| matches!(dt, DatumType::F16 | DatumType::F32)));
    Ok(Some(Box::new(tract_gpu::ops::rms_norm::GpuRmsNorm::new(
        op.axis,
        op.eps.clone(),
        true,
        false,
        op.out_dt,
        op.scale_dt == Some(DatumType::F16),
        "Wgpu",
        wgpu_rms_norm_dispatch,
    ))))
});
