use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::{MTLResourceUsage, MTLSize};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

/// Rows of the activation the ring matvec kernel serves.
pub const RING_MATVEC_MAX_M: usize = 8;
/// Columns a threadgroup covers, and the most slices of `k` it splits them over:
/// `RING_MATVEC_COLS` and `RING_MATVEC_SLICES` in the kernel.
const RING_MATVEC_COLS: usize = 32;
const RING_MATVEC_SLICES: usize = 8;
/// Rows of the window a slice walks at the least, below which a threadgroup
/// takes fewer slices.
const RING_MATVEC_ROWS_PER_SLICE: usize = 32;

/// The axis of a channel's `[k, n]` window a ring turns on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingAxis {
    /// The contracted axis, a slot being a run of whole rows.
    K,
    /// The output columns, a slot being a run of every row's columns.
    N,
}

/// `out[c, r, j] = sum_kk act[c, r, kk] * window(c, kk, j)` for `channels`
/// windows held in `ring`, element `(kk, j)` at `kk * k_stride + j * n_stride`
/// from the start `table` gives for channel `c`, along with how far its window
/// has turned along `axis`. `act` is `[channels, m, k]` and `out`
/// `[channels, m, n]`, both contiguous.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_ring_matvec(
    stream: &MetalStream,
    act: &DeviceTensor,
    ring: &DeviceTensor,
    out: &DeviceTensor,
    table: &DeviceTensor,
    (channels, m, k, n): (usize, usize, usize, usize),
    (k_stride, n_stride): (usize, usize),
    axis: RingAxis,
) -> TractResult<()> {
    let dt = act.datum_type();
    ensure!(ring.datum_type() == dt && out.datum_type() == dt);
    ensure!(
        m <= RING_MATVEC_MAX_M,
        "The ring matvec kernel serves up to {RING_MATVEC_MAX_M} rows, not {m}"
    );
    ensure!(act.len() == channels * m * k && out.len() == channels * m * n);
    ensure!(
        act.strides() == natural_strides(act.shape()).as_slice(),
        "The ring matvec reads a contiguous activation"
    );
    ensure!(table.len() == 2 * channels && table.datum_type() == i32::datum_type());
    let name = match dt {
        DatumType::F32 => "ring_matvec_f32",
        DatumType::F16 => "ring_matvec_f16",
        _ => bail!("No ring matvec kernel for {dt:?}"),
    };
    for t in [act, ring, out, table] {
        stream.retain_tensor(t);
    }
    let slices = (k / RING_MATVEC_ROWS_PER_SLICE).clamp(1, RING_MATVEC_SLICES);
    let pipeline = stream.load_pipeline(LibraryName::RingMatVec, name)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, act, MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, ring, MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, out, MTLResourceUsage::Write);
        encoder.set_metal_tensor(3, table, MTLResourceUsage::Read);
        encoder.set_slice(4, &[m as i32]);
        encoder.set_slice(5, &[k as i32]);
        encoder.set_slice(6, &[n as i32]);
        encoder.set_slice(7, &[(axis == RingAxis::K) as i32]);
        encoder.set_slice(8, &[k_stride as i32]);
        encoder.set_slice(9, &[n_stride as i32]);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n.div_ceil(RING_MATVEC_COLS) as u64,
                height: channels as u64,
                depth: 1,
            },
            MTLSize { width: RING_MATVEC_COLS as u64, height: slices as u64, depth: 1 },
        );
    });
    Ok(())
}
