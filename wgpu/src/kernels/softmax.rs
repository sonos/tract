use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, keys_for, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Softmax, &["softmax"], shader_f16)
}

fn reshape_to_rank_3(shape: &[usize], axis: usize) -> [usize; 3] {
    let outer: usize = shape[..axis].iter().product();
    let k = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    [outer.max(1), k, inner.max(1)]
}

pub fn wgpu_softmax_dispatch(
    input: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        ensure!(output.shape() == input.shape());
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::Softmax, dtype: dt },
            entry: EntryPoint::typed("softmax", dt),
        })?;
        let nd3 = reshape_to_rank_3(input.shape(), axis);
        let n = (nd3[0] * nd3[2]) as u64;
        if n == 0 || nd3[1] == 0 {
            return Ok(());
        }
        let in_buf = get_wgpu_buffer(input);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(LayoutKind::Unary, &[in_buf, out_buf], q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            nd3[0] as u32,
            nd3[1] as u32,
            nd3[2] as u32,
            0,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("softmax", &pipeline, &bg, dyn_off, n)
    })
}

register_wgpu_op!(tract_core::ops::nn::Softmax, |source, node, op| {
    rule_if!(matches!(
        source.node_input_facts(node.id)?[0].datum_type,
        DatumType::F32 | DatumType::F16
    ));
    rule_if!(op.quant_output_dt.is_none());
    rule_if!(op.kind == tract_core::ops::nn::SoftmaxKind::Softmax);
    Ok(Some(Box::new(tract_gpu::ops::softmax::GpuSoftmax::from_tract_core(
        op,
        "Wgpu",
        wgpu_softmax_dispatch,
    )?)))
});
