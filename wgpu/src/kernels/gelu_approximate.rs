use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ShaderDtype, gelu_approximate_module, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Tanh-form GELU for a contiguous f16 or f32 tensor. `fast_impl` uses `x^2`
/// inside the tanh; the canonical form uses `x^3`. The input is read from its
/// byte offset as a flat run of `len` elements.
pub fn wgpu_gelu_approximate_dispatch(
    fast_impl: bool,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let stem = if fast_impl { "gelu_approx_fast" } else { "gelu_approx" };
        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            ("gelu_approximate", dt, fast_impl),
            layout,
            EntryPoint::typed(stem, dt),
            || gelu_approximate_module(dt),
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
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch(stem, &pipeline, &bg, dyn_off, input.len() as u64)
    })
}

// GeluApproximate is an ElementWiseMiniOp, so this shares ElementWiseOp's
// TypeId with the parameter-free element-wise translator. That one declines it.
register_wgpu_op!(tract_core::ops::element_wise::ElementWiseOp, |source, node, op| {
    rule_if_some!(ew = op.0.downcast_ref::<tract_core::ops::nn::GeluApproximate>());
    rule_if!(matches!(
        source.node_input_facts(node.id)?[0].datum_type,
        DatumType::F32 | DatumType::F16
    ));
    Ok(Some(Box::new(tract_gpu::ops::gelu_approximate::GpuGeluApproximate::new(
        ew.fast_impl,
        "Wgpu",
        wgpu_gelu_approximate_dispatch,
    ))))
});
