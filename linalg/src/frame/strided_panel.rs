//! K-major f32 panel that the SME kernel reads in place.
//!
//! A packed f32 panel is `r` contiguous values per K, then the next K `r * 4`
//! bytes later. NCHW (and any layout whose mn axis already has element stride 1)
//! is that same picture with a different K step: one channel plane apart. The
//! kernel loads it with that stride and the pack copy does not run.
//!
//! A last panel shorter than `r` is predicated. The packed fallback still
//! zero-fills that tail and reports `valid == r`, so the kernel's full-width
//! loop stays legal. Layouts that are not mn-contiguous take that fallback.

use std::fmt::Display;
use std::sync::Arc;

use tract_data::internal::*;

use crate::WeightType;
use crate::mmm::{EagerPackedInput, MMMInputFormat, MMMInputValue, PackedMatrixStorage};
use crate::pack::PackedFormat;

/// What one SME `AddMatMul` passes as `pa` when the packing index is the
/// strided one. Three machine words, little-endian: data pointer, signed K
/// byte stride, valid lanes in `1..=r` (`r` means the tail is already zero).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PanelDesc {
    ptr: usize,
    k_byte_stride: usize,
    valid: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum KeepAlive {
    /// Bytes stay in the activation. `PanelDesc::ptr` points into this tensor.
    Tensor(Arc<Tensor>),
    /// mn was not contiguous. Same descriptor ABI, aimed at a packed buffer.
    Packed(Arc<Blob>),
}

/// f32 operand whose panels the 32×32 SME kernel can address without a pack.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct StridedKMajor {
    r: usize,
}

impl StridedKMajor {
    pub const fn new(r: usize) -> Self {
        Self { r }
    }

    /// Build the operand from a tensor the caller already owns.
    ///
    /// `byte_offset` selects one batch item. No activation bytes are copied
    /// when the mn axis is contiguous and every addressed element sits in the
    /// allocation; anything else is packed.
    pub fn panel_from_arc(
        &self,
        tensor: Arc<Tensor>,
        byte_offset: isize,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        ensure!(k_axis != mn_axis, "k and mn axes are both {k_axis}");
        ensure!(k_axis < tensor.rank() && mn_axis < tensor.rank());
        ensure!(tensor.datum_type() == f32::datum_type());
        let k = tensor.shape()[k_axis];
        let mn = tensor.shape()[mn_axis];
        let k_stride = tensor.strides()[k_axis];
        let mn_stride = tensor.strides()[mn_axis];
        if mn_stride == 1 && spans_allocation(&tensor, byte_offset, k, mn, k_stride) {
            Ok(Box::new(StridedPanelInput::view(*self, tensor, byte_offset, k, mn, k_stride)))
        } else {
            let view = unsafe {
                TensorView::from_bytes(&tensor, byte_offset, tensor.shape(), tensor.strides())
            };
            let packed = packed_format(self.r).pack_tensor_view(&view, k_axis, mn_axis)?;
            self.wrap_packed(packed)
        }
    }

    fn wrap_packed(&self, packed: Box<dyn MMMInputValue>) -> TractResult<Box<dyn MMMInputValue>> {
        let eager = packed
            .downcast::<EagerPackedInput>()
            .map_err(|_| format_err!("strided fallback did not pack to EagerPackedInput"))?;
        let panel_bytes = eager.panel_bytes;
        let k = eager.k();
        let mn = eager.mn();
        let blob = eager.packed.clone();
        let base = blob.as_bytes().as_ptr() as usize;
        let mut panels = Vec::with_capacity(mn.div_ceil(self.r).max(1));
        if mn == 0 {
            panels.push(PanelDesc { ptr: base, k_byte_stride: (self.r * 4), valid: 0 });
        } else {
            // The packer zero-fills a short tail, so every panel is a full `r`.
            for p in 0..mn.div_ceil(self.r) {
                panels.push(PanelDesc {
                    ptr: base + p * panel_bytes,
                    k_byte_stride: (self.r * 4),
                    valid: self.r,
                });
            }
        }
        Ok(Box::new(StridedPanelInput {
            format: *self,
            keep: KeepAlive::Packed(blob),
            fact: NoFact,
            panels,
            k,
            mn,
        }))
    }
}

fn packed_format(r: usize) -> PackedFormat {
    PackedFormat::new(f32::datum_type(), r, 128)
}

/// True when both ends of the K range and both ends of the mn range land
/// inside the allocation. A hostile stride falls back to packing.
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
    let elem = 4isize;
    let last_k = (k as isize) - 1;
    let last_mn = (mn as isize) - 1;
    for ik in [0, last_k] {
        for im in [0, last_mn] {
            // `byte_offset` is already bytes. The strides are elements.
            let elem_off = ik.saturating_mul(k_stride).saturating_add(im);
            let off = byte_offset.saturating_add(elem_off.saturating_mul(elem));
            if off < 0 || off.saturating_add(elem) > bytes {
                return false;
            }
        }
    }
    true
}

impl MMMInputFormat for StridedKMajor {
    fn prepare_tensor(&self, t: &Tensor, k_axis: usize, mn_axis: usize) -> TractResult<Tensor> {
        let packed = self.prepare_one(t, k_axis, mn_axis)?;
        Ok(PackedMatrixStorage::new(packed).into_tensor(t.datum_type()))
    }

    fn prepare_one(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        // Deep copy: this is the kernel-test entry, which hands us a borrow.
        // The graph path uses `panel_from_arc` and does not copy.
        self.panel_from_arc(Arc::new(t.clone()), 0, k_axis, mn_axis)
    }

    fn prepare_one_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        let packed = packed_format(self.r).pack_tensor_view(t, k_axis, mn_axis)?;
        self.wrap_packed(packed)
    }

    fn precursor(&self) -> WeightType {
        WeightType::Plain(f32::datum_type())
    }

    fn r(&self) -> usize {
        self.r
    }

    fn k_alignment(&self) -> usize {
        1
    }

    fn mem_size(&self, _k: TDim, _mn: TDim) -> TDim {
        0.to_dim()
    }

    fn extract_at_mn_f16(&self, _: &EagerPackedInput, _: usize, _: &mut [f16]) -> TractResult<()> {
        bail!("strided f32 panel has no f16 extract")
    }

    fn extract_at_mn_f32(&self, _: &EagerPackedInput, _: usize, _: &mut [f32]) -> TractResult<()> {
        bail!("strided f32 panel is not an EagerPackedInput")
    }
}

impl Display for StridedKMajor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StridedF32[{}]", self.r)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct NoFact;

impl ExoticFact for NoFact {
    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(0.to_dim())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StridedPanelInput {
    format: StridedKMajor,
    keep: KeepAlive,
    fact: NoFact,
    panels: Vec<PanelDesc>,
    k: usize,
    mn: usize,
}

impl StridedPanelInput {
    fn view(
        format: StridedKMajor,
        tensor: Arc<Tensor>,
        byte_offset: isize,
        k: usize,
        mn: usize,
        k_stride: isize,
    ) -> Self {
        let base = unsafe { tensor.as_ptr_unchecked::<u8>() } as isize + byte_offset;
        let k_byte_stride = k_stride * 4;
        let mut panels = Vec::with_capacity(mn.div_ceil(format.r).max(1));
        if mn == 0 {
            panels.push(PanelDesc {
                ptr: base as usize,
                k_byte_stride: k_byte_stride as usize,
                valid: 0,
            });
        } else {
            let mut left = mn;
            let mut off = 0isize;
            while left > 0 {
                let valid = left.min(format.r);
                panels.push(PanelDesc {
                    ptr: (base + off) as usize,
                    k_byte_stride: k_byte_stride as usize,
                    valid,
                });
                left -= valid;
                off += format.r as isize * 4;
            }
        }
        Self { format, keep: KeepAlive::Tensor(tensor), fact: NoFact, panels, k, mn }
    }

    pub fn is_view(&self) -> bool {
        matches!(self.keep, KeepAlive::Tensor(_))
    }
}

impl MMMInputValue for StridedPanelInput {
    fn format(&self) -> &dyn MMMInputFormat {
        &self.format
    }

    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        None
    }

    fn panel_bytes(&self, i: usize, _buffer: Option<*mut u8>) -> TractResult<*const u8> {
        self.panels
            .get(i)
            .map(|p| p as *const PanelDesc as *const u8)
            .with_context(|| format!("strided panel {i} of {}", self.panels.len()))
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

    fn extract_at_mn_f16(&self, _mn: usize, _slice: &mut [f16]) -> TractResult<()> {
        bail!("strided f32 panel has no f16 extract")
    }

    fn extract_at_mn_f32(&self, mn: usize, slice: &mut [f32]) -> TractResult<()> {
        ensure!(slice.len() == self.k);
        ensure!(mn < self.mn);
        let panel = mn / self.format.r;
        let lane = mn % self.format.r;
        let desc = &self.panels[panel];
        ensure!(lane < desc.valid);
        let stride = desc.k_byte_stride as isize;
        unsafe {
            let ptr = desc.ptr as *const u8;
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = *(ptr.offset(stride * i as isize).add(lane * 4) as *const f32);
            }
        }
        Ok(())
    }
}

impl Display for StridedPanelInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} k={} mn={} {} panels", self.format, self.k, self.mn, self.panels.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmm::MMMInputValue;

    fn words(input: &dyn MMMInputValue, panel: usize) -> (usize, usize, usize) {
        let p = input.panel_bytes(panel, None).unwrap() as *const usize;
        unsafe { (*p, *p.add(1), *p.add(2)) }
    }

    #[test]
    fn nchw_row_is_a_view() {
        let data: Vec<f32> = (0..3 * 40).map(|i| i as f32).collect();
        let tensor = Tensor::from_shape(&[3, 40], &data).unwrap();
        let start = tensor.as_bytes().as_ptr() as usize;
        let input = StridedKMajor::new(32).panel_from_arc(Arc::new(tensor), 0, 0, 1).unwrap();
        let view = input.downcast_ref::<StridedPanelInput>().unwrap();
        assert!(view.is_view());
        assert_eq!(input.panels_count(), 2);
        let (ptr, stride, valid) = words(&*input, 0);
        assert_eq!(stride, 40 * 4);
        assert_eq!(valid, 32);
        assert!(ptr >= start);
        assert_eq!(words(&*input, 1).2, 8);
        let mut got = [0f32; 3];
        input.extract_at_mn_f32(33, &mut got).unwrap();
        assert_eq!(got, [33.0, 73.0, 113.0]);
    }

    #[test]
    fn k_inner_falls_back_to_a_pack() {
        let data = vec![1f32; 32 * 4];
        let tensor = Tensor::from_shape(&[32, 4], &data).unwrap();
        let input = StridedKMajor::new(32).panel_from_arc(Arc::new(tensor), 0, 1, 0).unwrap();
        let view = input.downcast_ref::<StridedPanelInput>().unwrap();
        assert!(!view.is_view());
        let (_, stride, valid) = words(&*input, 0);
        assert_eq!(stride, 32 * 4);
        assert_eq!(valid, 32);
    }
}
