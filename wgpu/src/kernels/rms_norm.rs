use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, pack_u32s, rms_norm_module,
    rms_norm_residual_module,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Dispatch for `GpuRmsNorm`. `residual` and `sum_out` come and go together:
/// the kernel adds them in the input dtype, writes that sum, and normalizes it.
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
    ensure!(residual.is_some() == sum_out.is_some());
    ensure!(output.shape() == input.shape());
    ensure!(!round_scale_f16 || scale.is_some(), "round_scale_f16 requires a scale");
    let k = input.shape()[axis];
    if let Some(scale) = scale {
        ensure!(scale.datum_type() == DatumType::F32 && scale.len() == k);
    }
    if let (Some(residual), Some(sum_out)) = (residual, sum_out) {
        ensure!(residual.shape() == input.shape() && residual.datum_type() == input.datum_type());
        ensure!(sum_out.shape() == input.shape() && sum_out.datum_type() == input.datum_type());
    }
    let eps = eps.cast_to_scalar::<f32>()?;
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        if let Some(residual) = residual {
            q.retain_tensor(residual);
        }
        if let Some(sum_out) = sum_out {
            q.retain_tensor(sum_out);
        }
        let in_dt = ShaderDtype::from_datum(input.datum_type())?;
        let out_dt = ShaderDtype::from_datum(output.datum_type())?;
        let outer: usize = input.shape()[..axis].iter().product();
        let inner: usize = input.shape()[axis + 1..].iter().product();
        let rows = outer * inner;
        if rows == 0 || k == 0 {
            return Ok(());
        }
        let mut buffers = vec![get_wgpu_buffer(input)];
        let off_res = if let Some(residual) = residual {
            buffers.push(get_wgpu_buffer(residual));
            element_offset(residual, 0)
        } else {
            0
        };
        let mut off_scale = 0;
        if let Some(scale) = scale {
            q.retain_tensor(scale);
            buffers.push(get_wgpu_buffer(scale));
            off_scale = element_offset(scale, 0);
        }
        buffers.push(get_wgpu_buffer(output));
        let off_sum = if let Some(sum_out) = sum_out {
            buffers.push(get_wgpu_buffer(sum_out));
            element_offset(sum_out, 0)
        } else {
            0
        };
        let with_residual = residual.is_some();
        let scaled = scale.is_some();
        let layout = match (with_residual, scaled) {
            (false, false) => LayoutKind::Unary,
            (false, true) => LayoutKind::Chain(3),
            (true, false) => LayoutKind::TailWrites { n: 4, writes: 2 },
            (true, true) => LayoutKind::TailWrites { n: 5, writes: 2 },
        };
        let pipeline = q.context().chain_pipeline(
            ("rms_norm", in_dt, out_dt, scaled, round_scale_f16, with_residual),
            layout,
            EntryPoint::plain("rms_norm"),
            || {
                if with_residual {
                    rms_norm_residual_module(in_dt, out_dt, scaled, round_scale_f16)
                } else {
                    rms_norm_module(in_dt, out_dt, scaled, round_scale_f16)
                }
            },
        )?;
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let params = if with_residual {
            pack_u32s(&[
                element_offset(input, 0) as u32,
                off_res as u32,
                element_offset(output, 0) as u32,
                off_sum as u32,
                off_scale as u32,
                k as u32,
                inner as u32,
                rows as u32,
                eps.to_bits(),
            ])
        } else {
            pack_u32s(&[
                element_offset(input, 0) as u32,
                element_offset(output, 0) as u32,
                off_scale as u32,
                k as u32,
                inner as u32,
                rows as u32,
                eps.to_bits(),
            ])
        };
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
