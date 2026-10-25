use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::sdpa::Sdpa;

use crate::kernels::shaders::{
    EntryPoint, FLASH_SDPA_MAX_HEAD_DIM, FLASH_SDPA_ROWS, LayoutKind, ShaderDtype,
    flash_sdpa_module, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Whether the flash kernel takes this `Sdpa`: unmasked and not causal, f16 or
/// f32 throughout, rank 3 or 4 Q, K and V with the same leading dims, and
/// concrete head dims within [`FLASH_SDPA_MAX_HEAD_DIM`].
pub fn flash_sdpa_supported(facts: &[&TypedFact], op: &Sdpa) -> bool {
    if facts.len() != 3 || op.is_causal {
        return false;
    }
    let [qf, kf, vf] = [facts[0], facts[1], facts[2]];
    let rank = qf.rank();
    if !matches!(rank, 3 | 4)
        || kf.rank() != rank
        || vf.rank() != rank
        || !matches!(qf.datum_type, DatumType::F16 | DatumType::F32)
        || kf.datum_type != qf.datum_type
        || vf.datum_type != qf.datum_type
        || qf.shape[..rank - 2] != kf.shape[..rank - 2]
        || kf.shape[..rank - 1] != vf.shape[..rank - 1]
    {
        return false;
    }
    let dim = |f: &TypedFact| f.shape[rank - 1].to_usize().ok();
    match (dim(qf), dim(kf), dim(vf)) {
        (Some(d), Some(dk), Some(dv)) => {
            d == dk && d <= FLASH_SDPA_MAX_HEAD_DIM && dv <= FLASH_SDPA_MAX_HEAD_DIM
        }
        _ => false,
    }
}

/// (batch, head, position, channel) strides of a rank 3 or 4 attention operand.
fn bhsd_strides(t: &DeviceTensor) -> [u32; 4] {
    let s = t.strides();
    if s.len() == 3 {
        [s[0] as u32, 0, s[1] as u32, s[2] as u32]
    } else {
        [s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32]
    }
}

pub fn wgpu_flash_sdpa_dispatch(
    scale: f32,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|queue| {
        for t in [q, k, v, output] {
            queue.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(q.datum_type())?;
        let rank = q.rank();
        let (sq, d) = (q.shape()[rank - 2], q.shape()[rank - 1]);
        let (sk, dv) = (k.shape()[rank - 2], v.shape()[rank - 1]);
        let (batch, heads) =
            if rank == 4 { (q.shape()[0], q.shape()[1]) } else { (q.shape()[0], 1) };
        let layout = LayoutKind::Chain(4);
        let pipeline = queue.context().chain_pipeline(
            ("flash_sdpa", dt, d, dv),
            layout,
            EntryPoint::typed("flash_sdpa", dt),
            || flash_sdpa_module(dt, d, dv),
        )?;
        let buffers =
            [get_wgpu_buffer(q), get_wgpu_buffer(k), get_wgpu_buffer(v), get_wgpu_buffer(output)];
        let bg = queue.context().bind_group(layout, &buffers, queue.uniform())?;
        let mut vals = vec![
            element_offset(q, 0) as u32,
            element_offset(k, 0) as u32,
            element_offset(v, 0) as u32,
            element_offset(output, 0) as u32,
            sq as u32,
            sk as u32,
            scale.to_bits(),
            0,
        ];
        for t in [q, k, v, output] {
            vals.extend_from_slice(&bhsd_strides(t));
        }
        let dyn_off = queue.alloc_uniform(&pack_u32s(&vals))?;
        let grid = [sq.div_ceil(FLASH_SDPA_ROWS) as u32, heads as u32, batch as u32];
        queue.dispatch_grid("flash_sdpa", &pipeline, &bg, dyn_off, grid)
    })
}

register_wgpu_op!(Sdpa, |source, node, op| {
    rule_if!(flash_sdpa_supported(&source.node_input_facts(node.id)?, op));
    let d = source.node_input_facts(node.id)?[0].shape.last().unwrap().to_usize()?;
    let scale = match &op.scale {
        Some(s) => s.cast_to_scalar::<f32>()?,
        None => 1.0 / (d as f32).sqrt(),
    };
    Ok(Some(Box::new(crate::ops::sdpa::WgpuFlashSdpa { scale })))
});
