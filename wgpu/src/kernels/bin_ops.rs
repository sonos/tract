use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    BINARY_OPS, EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, Width,
    broadcast_strides, dtypes, op_in_set, pack_u32s, pad8_u32,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    let mut keys = Vec::new();
    for dt in dtypes(shader_f16) {
        for name in BINARY_OPS {
            for width in [Width::Scalar, Width::Vec4, Width::Vec4Splat] {
                keys.push(PipelineKey {
                    module: ModuleKey { kind: ModuleKind::Binary, dtype: *dt },
                    entry: EntryPoint::wide(name, width, *dt),
                });
            }
        }
    }
    keys
}

/// A wider thread only pays when the left operand runs with the output and the
/// right one either does too or is constant across the four values.
fn width_for(lhs: &DeviceTensor, rhs: &DeviceTensor, output: &DeviceTensor) -> Width {
    let out_shape = output.shape();
    let last = *out_shape.last().unwrap_or(&1);
    if !output.len().is_multiple_of(4) || !last.is_multiple_of(4) {
        return Width::Scalar;
    }
    let natural = Tensor::natural_strides(out_shape);
    let runs_with_output = |t: &DeviceTensor| t.shape() == out_shape && t.strides() == &natural[..];
    if !runs_with_output(lhs) {
        return Width::Scalar;
    }
    if runs_with_output(rhs) {
        Width::Vec4
    } else if rhs.rank() == out_shape.len() && rhs.shape().last() == Some(&1) && last != 1 {
        Width::Vec4Splat
    } else {
        Width::Scalar
    }
}

pub fn is_supported(mini_op: &dyn BinMiniOp, dt: DatumType) -> bool {
    let name = mini_op.name().to_lowercase();
    BINARY_OPS.contains(&name.as_str()) && matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn wgpu_bin_op_dispatch(
    mini_op: &dyn BinMiniOp,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(lhs);
        q.retain_tensor(rhs);
        q.retain_tensor(output);
        ensure!(lhs.rank() == rhs.rank());
        let dt = ShaderDtype::from_datum(lhs.datum_type())?;
        let name = op_in_set(BINARY_OPS, mini_op.name())
            .with_context(|| format!("tract-wgpu has no binary kernel for {}", mini_op.name()))?;
        let width = width_for(lhs, rhs, output);
        let key = PipelineKey {
            module: ModuleKey { kind: ModuleKind::Binary, dtype: dt },
            entry: EntryPoint::wide(name, width, dt),
        };
        let pipeline = q.context().pipeline(key)?;
        let lhs_buf = get_wgpu_buffer(lhs);
        let rhs_buf = get_wgpu_buffer(rhs);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(
            LayoutKind::Binary,
            &[lhs_buf, rhs_buf, out_buf],
            q.uniform(),
        )?;

        let out_shape = output.shape();
        let lhs_s = broadcast_strides(lhs.shape(), lhs.strides(), out_shape);
        let rhs_s = broadcast_strides(rhs.shape(), rhs.strides(), out_shape);
        let shp = pad8_u32(out_shape);
        let mut vals = vec![
            element_offset(lhs, 0) as u32,
            element_offset(rhs, 0) as u32,
            element_offset(output, 0) as u32,
            out_shape.len() as u32,
            output.len() as u32,
            0,
            0,
            0,
        ];
        vals.extend_from_slice(&lhs_s);
        vals.extend_from_slice(&rhs_s);
        vals.extend_from_slice(&shp);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        let threads = if width == Width::Scalar { output.len() } else { output.len() / 4 };
        q.dispatch("binary", &pipeline, &bg, dyn_off, threads as u64)
    })
}

pub fn wgpu_bin_op(mini_op: Box<dyn BinMiniOp>) -> tract_gpu::ops::binary::GpuBinOp {
    tract_gpu::ops::binary::GpuBinOp::new(mini_op, "Wgpu", wgpu_bin_op_dispatch)
}

register_wgpu_op!(tract_core::ops::binary::TypedBinOp, |source, node, op| {
    rule_if!(is_supported(&*op.0, source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(wgpu_bin_op(op.0.clone()))))
});
