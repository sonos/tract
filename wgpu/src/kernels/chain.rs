//! Dispatch for a fused elementwise chain.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    ChainOperand, ChainStep, EntryPoint, LayoutKind, ShaderDtype, broadcast_strides, chain_key,
    chain_wgsl, pack_u32s, pad8_u32,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

/// Extra operands a chain can read besides its head, bounded by what the
/// 256-byte uniform slot holds.
pub const MAX_EXTRA_INPUTS: usize = 3;

pub fn wgpu_chain_dispatch(
    steps: &[ChainStep],
    inputs: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        ensure!(inputs.len() <= MAX_EXTRA_INPUTS + 1, "chain has too many inputs");
        for t in inputs {
            q.retain_tensor(t);
        }
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(output.datum_type())?;
        let layout = LayoutKind::Chain(inputs.len() as u8 + 1);
        let out_shape = output.shape();
        let natural = Tensor::natural_strides(out_shape);
        let contiguous: Vec<bool> =
            inputs.iter().map(|t| t.shape() == out_shape && t.strides() == &natural[..]).collect();
        // Four values a thread: the kernel is bandwidth-bound, so a wider load
        // is the whole win. The four sit next to each other along the last
        // axis, so an operand that broadcasts over that axis is one value for
        // all of them and can be splatted; anything else still has to gather,
        // and then a wider thread buys nothing.
        let last = *out_shape.last().unwrap_or(&1);
        let kinds: Option<Vec<ChainOperand>> = (output.len().is_multiple_of(4)
            && last.is_multiple_of(4))
        .then(|| {
            inputs
                .iter()
                .zip(contiguous.iter())
                .map(|(t, c)| {
                    if *c {
                        Some(ChainOperand::Contiguous)
                    } else if t.rank() == out_shape.len()
                        && t.shape().last() == Some(&1)
                        && out_shape.last() != Some(&1)
                    {
                        Some(ChainOperand::Splat)
                    } else {
                        None
                    }
                })
                .collect::<Option<Vec<_>>>()
        })
        .flatten();
        let key = chain_key(dt, steps, &contiguous, kinds.as_deref());
        let pipeline =
            q.context().chain_pipeline(&key, layout, EntryPoint::plain("chain"), || {
                chain_wgsl(dt, steps, &contiguous, kinds.as_deref())
            })?;

        let mut buffers: Vec<&crate::context::WgpuBuffer> =
            inputs.iter().map(|t| get_wgpu_buffer(t)).collect();
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;

        let mut offsets = [0u32; 4];
        for (i, t) in inputs.iter().enumerate() {
            offsets[i] = element_offset(t, 0) as u32;
        }
        let mut vals =
            vec![element_offset(output, 0) as u32, output.len() as u32, out_shape.len() as u32, 0];
        vals.extend_from_slice(&offsets);
        vals.extend_from_slice(&pad8_u32(out_shape));
        for t in inputs {
            vals.extend_from_slice(&broadcast_strides(t.shape(), t.strides(), out_shape));
        }
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        let threads = if kinds.is_some() { output.len() / 4 } else { output.len() };
        q.dispatch("chain", &pipeline, &bg, dyn_off, threads as u64)
    })
}
