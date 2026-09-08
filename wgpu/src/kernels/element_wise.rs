use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseMiniOp;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    ELEMENT_WISE_OPS, EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype,
    keys_for, op_in_set, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::ElementWise, ELEMENT_WISE_OPS, shader_f16)
}

pub fn is_supported(mini_op: &dyn ElementWiseMiniOp, dt: DatumType) -> bool {
    let name = mini_op.name().to_lowercase();
    ELEMENT_WISE_OPS.contains(&name.as_str()) && matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn wgpu_element_wise_dispatch(
    mini_op: &dyn ElementWiseMiniOp,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let name = op_in_set(ELEMENT_WISE_OPS, &mini_op.name()).with_context(|| {
            format!("tract-wgpu has no element-wise kernel for {}", mini_op.name())
        })?;
        let key = PipelineKey {
            module: ModuleKey { kind: ModuleKind::ElementWise, dtype: dt },
            entry: EntryPoint::typed(name, dt),
        };
        let pipeline = q.context().pipeline(key)?;
        let in_buf = get_wgpu_buffer(input);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(LayoutKind::Unary, &[in_buf, out_buf], q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            output.len() as u32,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("element_wise", &pipeline, &bg, dyn_off, output.len() as u64)
    })
}

pub fn wgpu_element_wise_op(
    mini_op: Box<dyn ElementWiseMiniOp>,
) -> tract_gpu::ops::element_wise::GpuElementWise {
    tract_gpu::ops::element_wise::GpuElementWise::new(mini_op, "Wgpu", wgpu_element_wise_dispatch)
}

register_wgpu_op!(tract_core::ops::element_wise::ElementWiseOp, |source, node, op| {
    rule_if!(is_supported(&*op.0, source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(wgpu_element_wise_op(op.0.clone()))))
});
