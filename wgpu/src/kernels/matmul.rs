//! Naive prefix-broadcasting GEMM: one thread per output element.
//!
//! `PrefixMatMul` is what tract lowers `EinSum` to, and a MobileNet-style
//! segmenter reaches it through every pointwise convolution, so the backend
//! needs it even though the tiled kernels (tfjs `matmul_packed`, ORT
//! `subgroup_matrix_gemm`) are the ones worth having later.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    ChainStep, EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, keys_for,
    matmul_module, pack_u32s, program_key, rpad8_dims, rpad8_strides,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

/// Rank the 8 stride slots can address, two of them being the matmul axes.
pub const MAX_RANK: usize = 8;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::MatMul, &["matmul"], shader_f16)
}

/// `m`, `k`, `n` read off the trailing two axes of `a` and `b`.
pub fn mkn(
    a_shape: &[usize],
    b_shape: &[usize],
    transpose_a: bool,
    transpose_b: bool,
) -> TractResult<(usize, usize, usize)> {
    let rank = a_shape.len();
    ensure!(rank >= 2 && rank == b_shape.len(), "matmul wants two tensors of equal rank >= 2");
    let (m, k) = if transpose_a {
        (a_shape[rank - 1], a_shape[rank - 2])
    } else {
        (a_shape[rank - 2], a_shape[rank - 1])
    };
    let (kb, n) = if transpose_b {
        (b_shape[rank - 1], b_shape[rank - 2])
    } else {
        (b_shape[rank - 2], b_shape[rank - 1])
    };
    ensure!(k == kb, "matmul k mismatch: a says {k}, b says {kb}");
    Ok((m, k, n))
}

/// Output shape of `PrefixMatMul`: broadcast prefix, then `m`, `n` (swapped
/// when `transpose_c`).
pub fn output_shape(
    a_shape: &[usize],
    b_shape: &[usize],
    transpose_a: bool,
    transpose_b: bool,
    transpose_c: bool,
) -> TractResult<TVec<usize>> {
    let rank = a_shape.len();
    let (m, _, n) = mkn(a_shape, b_shape, transpose_a, transpose_b)?;
    let mut shape: TVec<usize> =
        (0..rank - 2).map(|ix| if a_shape[ix] == 1 { b_shape[ix] } else { a_shape[ix] }).collect();
    if transpose_c {
        shape.push(n);
        shape.push(m);
    } else {
        shape.push(m);
        shape.push(n);
    }
    Ok(shape)
}

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

/// Epilogue operands are a scalar or one value per output column.
pub fn epilogue_mode(extra: &DeviceTensor, n: usize) -> TractResult<u32> {
    match extra.len() {
        1 => Ok(0),
        len if len == n => Ok(1),
        len => bail!("epilogue operand of {len} values fits neither a scalar nor {n} columns"),
    }
}

/// The transpose flags a `PrefixMatMul` carries, as the kernel reads them.
#[derive(Debug, Clone, Copy)]
pub struct Transposes {
    pub a: bool,
    pub b: bool,
    pub c: bool,
}

pub fn wgpu_matmul_dispatch(
    t: Transposes,
    epilogue: &[ChainStep],
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    let (transpose_a, transpose_b, transpose_c) = (t.a, t.b, t.c);
    with_wgpu_queue(|q| {
        q.retain_tensor(a);
        q.retain_tensor(b);
        q.retain_tensor(output);
        ensure!(a.datum_type() == b.datum_type(), "matmul inputs must share a dtype");
        ensure!(a.rank() <= MAX_RANK, "tract-wgpu matmul is limited to rank {MAX_RANK}");
        for t in extras {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(a.datum_type())?;
        let (m, k, n) = mkn(a.shape(), b.shape(), transpose_a, transpose_b)?;
        let layout = LayoutKind::Chain(3 + extras.len() as u8);
        let entry = EntryPoint::typed("matmul", dt);
        let pipeline = if epilogue.is_empty() {
            q.context().pipeline(PipelineKey {
                module: ModuleKey { kind: ModuleKind::MatMul, dtype: dt },
                entry,
            })?
        } else {
            let key = program_key("matmul", dt, epilogue, extras.len());
            q.context()
                .chain_pipeline(&key, layout, entry, || matmul_module(dt, epilogue, extras.len()))?
        };
        let out_shape = output.shape();
        ensure!(
            out_shape
                == &*output_shape(a.shape(), b.shape(), transpose_a, transpose_b, transpose_c)?,
            "matmul output shape {out_shape:?} does not match its inputs"
        );
        let prefix: usize = out_shape[..out_shape.len() - 2].iter().product();

        let mut buffers: Vec<&crate::context::WgpuBuffer> =
            vec![get_wgpu_buffer(a), get_wgpu_buffer(b)];
        buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;

        let mut vals = vec![
            element_offset(a, 0) as u32,
            element_offset(b, 0) as u32,
            element_offset(output, 0) as u32,
            prefix as u32,
            m as u32,
            k as u32,
            n as u32,
            transpose_a as u32,
            transpose_b as u32,
            transpose_c as u32,
            0,
            0,
        ];
        vals.extend_from_slice(&rpad8_strides(a.shape(), a.strides(), out_shape));
        vals.extend_from_slice(&rpad8_strides(b.shape(), b.strides(), out_shape));
        vals.extend_from_slice(&rpad8_strides(out_shape, output.strides(), out_shape));
        vals.extend_from_slice(&rpad8_dims(out_shape));
        let mut off_extra = [0u32; 4];
        let mut mode_extra = [0u32; 4];
        for (i, t) in extras.iter().enumerate() {
            off_extra[i] = element_offset(t, 0) as u32;
            mode_extra[i] = epilogue_mode(t, n)?;
        }
        vals.extend_from_slice(&off_extra);
        vals.extend_from_slice(&mode_extra);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        q.dispatch("matmul", &pipeline, &bg, dyn_off, (prefix * m * n) as u64)
    })
}
