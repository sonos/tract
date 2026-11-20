//! Matmul operand packed one panel at a time, when the kernel asks for it.
//!
//! An eager pack reads the whole operand and writes the whole packed copy before the
//! matmul runs, and the kernel then reads that copy back. [`LazyPackedInput`] keeps a
//! reference to the source tensor instead and packs panel `i` into the caller's scratch
//! buffer from [`MMMInputValue::panel_bytes`], so the packed bytes stay in cache between
//! the pack and the kernel. Packed panels are byte-identical to the eager
//! [`PackedFormat`] ones.

use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use tract_data::internal::*;

use crate::mmm::{MMMInputFormat, MMMInputValue, PackedExoticFact};
use crate::pack::PackedFormat;

/// One batch item of `tensor`, packed lazily to `format`.
///
/// Only built over a source whose mn axis has element stride 1 and whose every addressed
/// element lies inside the allocation: `panel_bytes` reads it unchecked.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct LazyPackedInput {
    fact: PackedExoticFact,
    format: PackedFormat,
    tensor: Arc<Tensor>,
    byte_offset: isize,
    k: usize,
    mn: usize,
    k_stride: isize,
}

impl LazyPackedInput {
    /// Wrap one `k_axis` x `mn_axis` matmul of `tensor`, starting `byte_offset` bytes in, for
    /// `format`. None when the mn axis is not contiguous or the addressed range does not fit
    /// the allocation: the caller packs eagerly then.
    pub fn new(
        format: &PackedFormat,
        tensor: Arc<Tensor>,
        byte_offset: isize,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Option<LazyPackedInput>> {
        ensure!(k_axis != mn_axis, "k and mn axes are both {k_axis}");
        ensure!(k_axis < tensor.rank() && mn_axis < tensor.rank());
        ensure!(
            tensor.datum_type().unquantized() == format.dt.unquantized(),
            "Attempting to lazily pack {:?} tensor for {format}",
            tensor.datum_type()
        );
        if !tensor.is_plain_ram() {
            return Ok(None);
        }
        let k = tensor.shape()[k_axis];
        let mn = tensor.shape()[mn_axis];
        let k_stride = tensor.strides()[k_axis];
        if tensor.strides()[mn_axis] != 1
            || !spans_allocation(&tensor, byte_offset, k, mn, k_stride)
        {
            return Ok(None);
        }
        let fact = PackedExoticFact { format: Box::new(format.clone()), mn: mn.to_dim(), k };
        Ok(Some(LazyPackedInput {
            fact,
            format: format.clone(),
            tensor,
            byte_offset,
            k,
            mn,
            k_stride,
        }))
    }

    fn source_at(&self, k: usize, mn: usize) -> TractResult<f32> {
        let dt = self.tensor.datum_type();
        let size = dt.size_of() as isize;
        let at = self.byte_offset + (k as isize * self.k_stride + mn as isize) * size;
        let bytes = &self.tensor.as_bytes()[at as usize..][..size as usize];
        Ok(match dt.unquantized() {
            DatumType::F16 => f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
            DatumType::F32 => f32::from_le_bytes(bytes.try_into()?),
            other => bail!("no extraction from lazily packed {other:?}"),
        })
    }
}

/// True when both ends of the k range and both ends of the mn range land inside the
/// tensor's allocation.
fn spans_allocation(
    tensor: &Tensor,
    byte_offset: isize,
    k: usize,
    mn: usize,
    k_stride: isize,
) -> bool {
    if k == 0 || mn == 0 {
        return true;
    }
    let bytes = tensor.as_bytes().len() as isize;
    let elem = tensor.datum_type().size_of() as isize;
    [0, k as isize - 1].iter().all(|&ik| {
        [0, mn as isize - 1].iter().all(|&im| {
            let off = ik
                .checked_mul(k_stride)
                .and_then(|o| o.checked_add(im))
                .and_then(|o| o.checked_mul(elem))
                .and_then(|o| o.checked_add(byte_offset));
            off.is_some_and(|off| off >= 0 && off + elem <= bytes)
        })
    })
}

impl MMMInputValue for LazyPackedInput {
    fn format(&self) -> &dyn MMMInputFormat {
        &self.format
    }

    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        Some(self.format.single_panel_layout(self.k, self.format.dt.size_of()))
    }

    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        let scratch = buffer.context("lazy packing needs a scratch panel buffer")?;
        let r = self.format.r;
        ensure!(
            i * r < self.mn.max(1),
            "panel {i} out of {} (mn={})",
            self.panels_count(),
            self.mn
        );
        let dt = self.tensor.datum_type();
        unsafe {
            if (i + 1) * r > self.mn {
                // The kernel computes on the padding lanes before their results are
                // discarded; garbage there decodes to denormals and stalls the fp pipeline.
                let layout = self.scratch_panel_buffer_layout().unwrap();
                std::ptr::write_bytes(scratch, 0, layout.size());
            }
            let src = self.tensor.as_ptr_unchecked::<u8>().offset(self.byte_offset);
            dispatch_copy!(PackedFormat::pack_t(dt)(
                &self.format,
                scratch as _,
                src as *const _,
                self.mn,
                self.k_stride,
                1,
                0..self.k,
                i * r..(i + 1) * r
            ));
        }
        Ok(scratch)
    }

    fn mn(&self) -> usize {
        self.mn
    }

    fn k(&self) -> usize {
        self.k
    }

    fn exotic_fact(&self) -> &dyn ExoticFact {
        &self.fact
    }

    fn extract_at_mn_f16(&self, mn: usize, slice: &mut [f16]) -> TractResult<()> {
        ensure!(slice.len() == self.k && mn < self.mn);
        for (k, slot) in slice.iter_mut().enumerate() {
            *slot = f16::from_f32(self.source_at(k, mn)?);
        }
        Ok(())
    }

    fn extract_at_mn_f32(&self, mn: usize, slice: &mut [f32]) -> TractResult<()> {
        ensure!(slice.len() == self.k && mn < self.mn);
        for (k, slot) in slice.iter_mut().enumerate() {
            *slot = self.source_at(k, mn)?;
        }
        Ok(())
    }
}

impl Display for LazyPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lazy {} (k={} mn={})", self.format, self.k, self.mn)
    }
}

impl Debug for LazyPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panels_match_eager(dt: DatumType, r: usize, shape: &[usize], k_axis: usize, mn_axis: usize) {
        let len = shape.iter().product::<usize>();
        let tensor =
            Tensor::from_shape(shape, &(0..len).map(|i| i as f32 * 0.25).collect::<Vec<_>>())
                .unwrap()
                .cast_to_dt(dt)
                .unwrap()
                .into_owned();
        let format = PackedFormat::new(dt, r, 16);
        let eager = format.pack_tensor(&tensor, k_axis, mn_axis).unwrap();
        let lazy = LazyPackedInput::new(&format, Arc::new(tensor), 0, k_axis, mn_axis)
            .unwrap()
            .expect("contiguous mn should be lazy");
        assert_eq!(lazy.panels_count(), eager.panels_count());
        let layout = lazy.scratch_panel_buffer_layout().unwrap();
        let panel_bytes = layout.size();
        let mut scratch = unsafe { Blob::for_layout(layout) };
        for i in 0..lazy.panels_count() {
            let p = lazy.panel_bytes(i, Some(scratch.as_mut_ptr())).unwrap();
            let e = eager.panel_bytes(i, None).unwrap();
            let k_bytes = lazy.k() * r * dt.size_of();
            unsafe {
                assert_eq!(
                    std::slice::from_raw_parts(p, k_bytes),
                    std::slice::from_raw_parts(e, k_bytes),
                    "panel {i} of {shape:?}"
                );
            }
            assert!(k_bytes <= panel_bytes);
        }
        let mut col = vec![0f32; lazy.k()];
        let mut expected = vec![0f32; lazy.k()];
        for mn in [0, lazy.mn() - 1] {
            lazy.extract_at_mn_f32(mn, &mut col).unwrap();
            eager.extract_at_mn_f32(mn, &mut expected).unwrap();
            assert_eq!(col, expected);
        }
    }

    #[test]
    fn nchw_panels_match_eager_pack() {
        panels_match_eager(f16::datum_type(), 16, &[48, 400], 0, 1);
        panels_match_eager(f32::datum_type(), 8, &[3, 7, 100], 1, 2);
    }

    #[test]
    fn partial_last_panel_is_zero_padded() {
        panels_match_eager(f16::datum_type(), 16, &[5, 37], 0, 1);
        panels_match_eager(f32::datum_type(), 16, &[9, 3], 0, 1);
    }

    #[test]
    fn non_contiguous_mn_is_not_lazy() {
        let tensor = Tensor::zero::<f32>(&[64, 8]).unwrap();
        let format = PackedFormat::new(f32::datum_type(), 8, 16);
        assert!(LazyPackedInput::new(&format, Arc::new(tensor), 0, 1, 0).unwrap().is_none());
    }
}
