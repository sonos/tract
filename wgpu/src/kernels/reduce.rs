use tract_core::internal::*;
use tract_gpu::ops::reduce::Reducer;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, keys_for, op_in_set,
    pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// The reducers the module has an entry point for, spelled as the WGSL names
/// them.
const REDUCE_OPS: &[&str] =
    &["reduce_sum", "reduce_prod", "reduce_max", "reduce_min", "reduce_mean_of_squares"];

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Reduce, REDUCE_OPS, shader_f16)
}

fn reshape_to_rank_3(shape: &[usize], axis: usize) -> [usize; 3] {
    let outer: usize = shape[..axis].iter().product();
    let k = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    [outer.max(1), k, inner.max(1)]
}

pub fn wgpu_reduce_launch(
    reducer: &Reducer,
    input: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        ensure!(output.datum_type() == input.datum_type());
        ensure!(output.shape()[axis] == 1);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        if reducer.is_logic() {
            bail!("tract-wgpu reduce has no bool path yet");
        }
        let name = op_in_set(REDUCE_OPS, &format!("reduce_{reducer}"))
            .with_context(|| format!("tract-wgpu has no reduce kernel for {reducer}"))?;
        let key = PipelineKey {
            module: ModuleKey { kind: ModuleKind::Reduce, dtype: dt },
            entry: EntryPoint::typed(name, dt),
        };
        let pipeline = q.context().pipeline(key)?;
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
        q.dispatch("reduce", &pipeline, &bg, dyn_off, n)
    })
}

register_wgpu_op!(tract_core::ops::nn::Reduce, |source, node, op| {
    let dt = source.node_input_facts(node.id)?[0].datum_type;
    if let Ok(gpu_op) =
        tract_gpu::ops::reduce::GpuReduce::from_tract_core(op, "Wgpu", wgpu_reduce_launch)
    {
        rule_if!(gpu_op.reducer.is_supported_dt(dt));
        return Ok(Some(Box::new(gpu_op)));
    }
    Ok(None)
});
