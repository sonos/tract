//! Dispatch for two chained matrix-vector products.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::matmul::epilogue_mode;
use crate::kernels::shaders::{
    ChainStep, EntryPoint, LayoutKind, ShaderDtype, gemv_pair_module, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

/// The widest hidden vector the kernel will take: it is held in workgroup
/// memory, one slot per value.
pub const MAX_WIDTH: usize = crate::kernels::shaders::GEMV_PAIR_WG as usize;

/// Operands the activation between the products can carry — a bias and a clamp
/// — bounded by the one `vec4` of offsets the uniform holds.
pub const MAX_ACT_OPERANDS: usize = 4;

/// `x` is one row of `k` values, `w1` is `k` by `r` and `w2` is `r` by `n`.
/// Strides are in elements, row first.
#[derive(Debug, Clone, Copy)]
pub struct GemvPairShape {
    pub k: usize,
    pub r: usize,
    pub n: usize,
    pub x_s: usize,
    pub w1: (usize, usize),
    pub w2: (usize, usize),
    pub out_s: usize,
}

pub fn wgpu_gemv_pair_dispatch(
    shape: GemvPairShape,
    act: &[ChainStep],
    x: &DeviceTensor,
    w1: &DeviceTensor,
    w2: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        for t in [x, w1, w2, output] {
            q.retain_tensor(t);
        }
        for t in extras {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(x.datum_type())?;
        let layout = LayoutKind::Chain(4 + extras.len() as u8);
        let key = format!(
            "gemv_pair_{}_{}_{}",
            dt.suffix(),
            extras.len(),
            act.iter().map(|s| format!("{s:?}")).collect::<String>()
        );
        let pipeline =
            q.context().chain_pipeline(&key, layout, EntryPoint::typed("gemv_pair", dt), || {
                gemv_pair_module(dt, act, extras.len())
            })?;
        let mut buffers = vec![get_wgpu_buffer(x), get_wgpu_buffer(w1), get_wgpu_buffer(w2)];
        buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let mut vals = vec![
            element_offset(x, 0) as u32,
            element_offset(w1, 0) as u32,
            element_offset(w2, 0) as u32,
            element_offset(output, 0) as u32,
            shape.k as u32,
            shape.r as u32,
            shape.n as u32,
            shape.x_s as u32,
            shape.w1.0 as u32,
            shape.w1.1 as u32,
            shape.w2.0 as u32,
            shape.w2.1 as u32,
            shape.out_s as u32,
            0,
            0,
            0,
        ];
        let mut off_extra = [0u32; 4];
        let mut mode_extra = [0u32; 4];
        for (i, t) in extras.iter().enumerate() {
            off_extra[i] = element_offset(t, 0) as u32;
            mode_extra[i] = epilogue_mode(t, shape.r)?;
        }
        vals.extend_from_slice(&off_extra);
        vals.extend_from_slice(&mode_extra);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        q.dispatch_grid("gemv_pair", &pipeline, &bg, dyn_off, [1, 1, 1])
    })
}
