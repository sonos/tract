use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    if !shader_f16 {
        return vec![];
    }
    vec![
        PipelineKey {
            module: ModuleKey { kind: ModuleKind::Cast, dtype: ShaderDtype::F32 },
            entry: EntryPoint::plain("cast_f32_f16"),
        },
        PipelineKey {
            module: ModuleKey { kind: ModuleKind::Cast, dtype: ShaderDtype::F16 },
            entry: EntryPoint::plain("cast_f16_f32"),
        },
    ]
}

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn wgpu_cast_dispatch(input: &DeviceTensor, output: &DeviceTensor) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        ensure!(input.shape() == output.shape());
        let (dtype, entry) = match (input.datum_type(), output.datum_type()) {
            (DatumType::F32, DatumType::F16) => {
                (ShaderDtype::F32, EntryPoint::plain("cast_f32_f16"))
            }
            (DatumType::F16, DatumType::F32) => {
                (ShaderDtype::F16, EntryPoint::plain("cast_f16_f32"))
            }
            (a, b) if a == b => return Ok(()),
            (a, b) => bail!("tract-wgpu cast {a:?} -> {b:?} not implemented"),
        };
        let pipeline = q
            .context()
            .pipeline(PipelineKey { module: ModuleKey { kind: ModuleKind::Cast, dtype }, entry })?;
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
        q.dispatch("cast", &pipeline, &bg, dyn_off, output.len() as u64)
    })
}

register_wgpu_op!(tract_core::ops::cast::Cast, |_source, _node, op| {
    Ok(tract_gpu::ops::cast::GpuCast::new(op.to, "Wgpu", wgpu_cast_dispatch, is_supported_dt)
        .map(|c| Box::new(c) as _))
});
