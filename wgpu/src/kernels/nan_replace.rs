use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{EntryPoint, LayoutKind, ShaderDtype, nan_replace_module, pack_u32s};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn wgpu_nan_replace_dispatch(
    value: f32,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            ("nan_replace", dt),
            layout,
            EntryPoint::typed("nan_replace", dt),
            || nan_replace_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            input.len() as u32,
            value.to_bits(),
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("nan_replace", &pipeline, &bg, dyn_off, input.len() as u64)
    })
}
