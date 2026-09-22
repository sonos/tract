use tract_core::internal::*;
use tract_core::ops::cnn::PoolSpec;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, keys_for, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    Max,
    Sum { count_include_pad: bool, normalize: bool },
}

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Pool, &["max_pool_2d", "sum_pool_2d"], shader_f16)
}

pub fn wgpu_pool_supported(pool_spec: &PoolSpec, fact: &TypedFact) -> bool {
    matches!(fact.datum_type, DatumType::F16 | DatumType::F32)
        && fact.rank() == 4
        && pool_spec.kernel_shape.len() == 2
        && fact.shape.as_concrete().is_some()
}

pub fn wgpu_pool_dispatch(
    pool_spec: &PoolSpec,
    kind: PoolKind,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let stem = match kind {
            PoolKind::Max => "max_pool_2d",
            PoolKind::Sum { .. } => "sum_pool_2d",
        };
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::Pool, dtype: dt },
            entry: EntryPoint::typed(stem, dt),
        })?;
        let in_shape = pool_spec.data_format.shape(input.shape())?;
        let out_shape = pool_spec.data_format.shape(output.shape())?;
        ensure!(in_shape.hw_rank() == 2);
        let strides = pool_spec.strides();
        let dilations = pool_spec.dilations();
        let padding = pool_spec.computed_padding(in_shape.hw_dims());
        let (count_include_pad, normalize) = match kind {
            PoolKind::Max => (0u32, 0u32),
            PoolKind::Sum { count_include_pad, normalize } => {
                (count_include_pad as u32, normalize as u32)
            }
        };
        let channels_last = pool_spec.data_format.c_is_last() as u32;
        let n = *in_shape.n().unwrap_or(&1);
        let c = *in_shape.c();
        let oh = out_shape.hw_dims()[0];
        let ow = out_shape.hw_dims()[1];
        let nelem = (n * oh * ow * c) as u64;
        let in_buf = get_wgpu_buffer(input);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(LayoutKind::Unary, &[in_buf, out_buf], q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            n as u32,
            in_shape.hw_dims()[0] as u32,
            in_shape.hw_dims()[1] as u32,
            c as u32,
            oh as u32,
            ow as u32,
            pool_spec.kernel_shape[0] as u32,
            pool_spec.kernel_shape[1] as u32,
            strides[0] as u32,
            strides[1] as u32,
            padding[0].pad_before as u32,
            padding[1].pad_before as u32,
            dilations[0] as u32,
            dilations[1] as u32,
            count_include_pad,
            normalize,
            channels_last,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("pool", &pipeline, &bg, dyn_off, nelem)
    })
}
