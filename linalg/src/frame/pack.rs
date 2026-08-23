use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Range;
use tract_data::internal::*;

use crate::mmm::{
    EagerPackedInput, MMMInputFormat, MMMInputValue, PackedExoticFact, PackedMatrixStorage,
};

use crate::WeightType;

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct PackedFormat {
    pub dt: DatumType,
    pub r: usize,
    pub alignment_bytes: usize,
    pub end_padding_record: usize,
}

impl MMMInputFormat for PackedFormat {
    fn prepare_tensor(&self, t: &Tensor, k_axis: usize, mn_axis: usize) -> TractResult<Tensor> {
        let packed = PackedFormat::pack_tensor(self, t, k_axis, mn_axis)?;
        Ok(PackedMatrixStorage::new(packed).into_tensor(t.datum_type()))
    }
    fn prepare_one_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        PackedFormat::pack_tensor_view(self, t, k_axis, mn_axis)
    }

    fn prepare_one(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        PackedFormat::pack_tensor(self, t, k_axis, mn_axis)
    }

    fn precursor(&self) -> WeightType {
        WeightType::Plain(self.dt)
    }

    fn r(&self) -> usize {
        self.r
    }

    fn k_alignment(&self) -> usize {
        1
    }

    #[allow(clippy::collapsible_if)]
    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        other: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        if let Some(other) = other.downcast_ref::<PackedFormat>() {
            if self.r == other.r && self.dt == other.dt {
                if self.alignment_bytes % other.alignment_bytes == 0
                    && self.end_padding_record >= other.end_padding_record
                {
                    return Some(self);
                }
                if other.alignment_bytes % self.alignment_bytes == 0
                    && other.end_padding_record >= self.end_padding_record
                {
                    return Some(other);
                }
            }
        }
        None
    }

    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        self.len(k, mn) * self.dt.size_of()
    }

    fn extract_at_mn_f16(
        &self,
        data: &EagerPackedInput,
        mn: usize,
        slice: &mut [f16],
    ) -> TractResult<()> {
        ensure!(data.format().dyn_eq(self));
        ensure!(self.len(data.k(), data.mn()) * self.dt.size_of() == data.packed.len());
        unsafe {
            let ptr = data.packed.as_ptr().add(
                (self.single_panel_len(data.k()) * (mn / self.r) + mn % self.r) * self.dt.size_of(),
            );
            for (i, slot) in slice.iter_mut().enumerate() {
                let ptr = ptr.add(i * self.dt.size_of() * self.r);
                *slot = if self.dt == f16::datum_type() {
                    *(ptr as *const f16)
                } else if self.dt == f32::datum_type() {
                    f16::from_f32(*(ptr as *const f32))
                } else {
                    bail!("Unexpected DT {:?}", self.dt)
                }
            }
        }
        Ok(())
    }

    fn extract_at_mn_f32(
        &self,
        data: &EagerPackedInput,
        mn: usize,
        slice: &mut [f32],
    ) -> TractResult<()> {
        ensure!(data.format().dyn_eq(self));
        ensure!(self.len(data.k(), data.mn()) * self.dt.size_of() == data.packed.len());
        unsafe {
            let ptr = data.packed.as_ptr().add(
                (self.single_panel_len(data.k()) * (mn / self.r) + mn % self.r) * self.dt.size_of(),
            );
            for (i, slot) in slice.iter_mut().enumerate() {
                let ptr = ptr.add(i * self.dt.size_of() * self.r);
                *slot = if self.dt == f16::datum_type() {
                    (*(ptr as *const f16)).to_f32()
                } else if self.dt == f32::datum_type() {
                    *(ptr as *const f32)
                } else {
                    bail!("Unexpected DT {:?}", self.dt)
                }
            }
        }
        Ok(())
    }
}

impl Display for PackedFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Packed{:?}[{}]", self.dt, self.r)
    }
}

impl Debug for PackedFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Packed{:?}[{}]@{}+{}",
            self.dt, self.r, self.alignment_bytes, self.end_padding_record
        )
    }
}

impl PackedFormat {
    pub const fn new(dt: DatumType, nr: usize, alignment_bytes: usize) -> PackedFormat {
        PackedFormat { dt, r: nr, alignment_bytes, end_padding_record: 1 }
    }

    pub const fn with_end_padding_record(self, end_padding_record: usize) -> Self {
        PackedFormat { end_padding_record, ..self }
    }

    #[inline]
    pub fn align(self, alignment: usize) -> Self {
        Self { alignment_bytes: alignment, ..self }
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment_bytes
    }

    #[inline]
    pub fn panel_width(&self) -> usize {
        self.r
    }

    #[inline]
    pub fn len<D: DimLike>(&self, k: D, n: D) -> D {
        n.divceil(self.r) * self.single_panel_len(k)
    }

    #[inline]
    pub fn single_panel_len<D: DimLike>(&self, k: D) -> D {
        ((k + self.end_padding_record) * self.r).divceil(self.alignment()) * self.alignment()
    }

    #[inline]
    pub fn single_panel_layout(&self, k: usize, item_size: usize) -> Layout {
        Layout::from_size_align(self.single_panel_len(k) * item_size, self.alignment()).unwrap()
    }

    pub fn pack_tensor(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        ensure!(t.datum_type().is_copy());
        self.pack_tensor_view(&t.view(), k_axis, mn_axis)
    }

    pub fn pack_tensor_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        ensure!(
            t.datum_type().unquantized() == self.dt.unquantized(),
            "Attempting to pack for {self} tensor view {t:?}"
        );
        let k = t.shape()[k_axis];
        let mn = t.shape()[mn_axis];
        let packed_len = self.len(k, mn);
        let panel_len = self.single_panel_len(k);
        let panel_bytes = panel_len * t.datum_type().size_of();
        let strides = t.strides();
        unsafe {
            let mut packed = Blob::new_for_size_and_align(
                t.datum_type().size_of() * packed_len,
                self.alignment_bytes,
            );
            if cfg!(debug_assertions) {
                packed.as_bytes_mut().fill(0u8);
            } else if mn % self.r != 0 {
                // The kernel computes on the last panel's padding lanes before
                // their results are discarded; garbage bytes there decode to
                // denormals and stall the fp pipeline. Zero the partial panel.
                packed.as_bytes_mut()[(mn / self.r) * panel_bytes..].fill(0u8);
            }
            dispatch_copy!(Self::pack_t(t.datum_type())(
                self,
                packed.as_mut_ptr() as _,
                t.as_ptr_unchecked(),
                mn,
                strides[k_axis],
                strides[mn_axis],
                0..k,
                0..mn
            ));
            Ok(Box::new(EagerPackedInput {
                fact: PackedExoticFact { format: Box::new(self.clone()), mn: mn.to_dim(), k },
                packed: packed.into(),
                panel_bytes,
                mn,
            }))
        }
    }

    pub unsafe fn pack<'a, 'b>(
        &self,
        pb: impl std::borrow::BorrowMut<TensorView<'a>>,
        b: impl std::borrow::Borrow<TensorView<'b>>,
        k_axis: usize,
        mn_axis: usize,
    ) {
        let k = b.borrow().shape()[k_axis];
        let mn = b.borrow().shape()[mn_axis];
        unsafe { self.pack_segment(pb, b, k_axis, mn_axis, 0..k, 0..mn) };
    }


    #[allow(clippy::too_many_arguments)]
    #[rustfmt::skip]
    pub unsafe fn pack_t<T: Datum + Copy>(
        &self,
        pb: *mut T,
        b: *const T,
        mn: usize,
        k_stride: isize,
        mn_stride: isize,
        k_range: Range<usize>,
        mn_range: Range<usize>,
        ) { unsafe {
        if k_range.len() == 0 || mn_range.len() == 0 {
            return
        }
        if self.r == 1 && k_stride == 1 && mn == 1 {
            pb.copy_from_nonoverlapping(b.add(k_range.start), k_range.len())
        } else if mn_stride == 1 {
            let size_of = T::datum_type().size_of();
            let rbytes = self.r * size_of;
            let mn_valid_end = mn_range.end.min(mn);
            let mn_range_bytes = mn_range.start * size_of..mn_valid_end * size_of;
            let k_stride_bytes = k_stride * size_of as isize;
            let bb = b as *const u8;
            let pbb = pb as *mut u8;
            let panel_len = self.single_panel_len(k_range.len()) * size_of;
            match rbytes {
                16 => pack_mn_major::<[u8; 16]>(bb, pbb, panel_len, k_stride_bytes, mn_range_bytes, k_range),
                24 => pack_mn_major::<[u8; 24]>(bb, pbb, panel_len, k_stride_bytes, mn_range_bytes, k_range),
                32 => pack_mn_major::<[u8; 32]>(bb, pbb, panel_len, k_stride_bytes, mn_range_bytes, k_range),
                48 => pack_mn_major::<[u8; 48]>(bb, pbb, panel_len, k_stride_bytes, mn_range_bytes, k_range),
                64 => pack_mn_major::<[u8; 64]>(bb, pbb, panel_len, k_stride_bytes, mn_range_bytes, k_range),
                96 => pack_mn_major::<[u8; 96]>(bb, pbb, panel_len, k_stride_bytes, mn_range_bytes, k_range),
                128 => pack_mn_major::<[u8; 128]>(bb, pbb, panel_len, k_stride_bytes, mn_range_bytes, k_range),
                _ => {
                    let mut packer = self.write_with_k_outer(pb, k_range.len(), mn_range.len());
                    for k in k_range {
                        for x in mn_range.start..mn_valid_end {
                            packer.write(*b.offset(x as isize + k_stride * k as isize))
                        }
                        for _x in mn_valid_end..mn_range.end {
                            packer.write(T::default())
                        }
                    }
                }
            }
        } else if k_stride == 1 {
            // just ignore invalid mn_range
            let mn_valid_end = mn_range.end.min(mn);
            if mn_valid_end > mn_range.start {
                pack_k_major(
                    b.offset(mn_range.start as isize * mn_stride + k_range.start as isize),
                    pb,
                    self.single_panel_len(k_range.len()),
                    self.r,
                    mn_stride,
                    k_range.len(),
                    mn_valid_end - mn_range.start,
                )
            }
        } else {
            let mut packer = self.write_with_k_outer(pb, k_range.len(), mn);
            let mn_valid_end = mn_range.end.min(mn);
            for k in k_range {
                for x in mn_range.start..mn_valid_end {
                    packer.write(*b.offset(x as isize * mn_stride + k_stride * k as isize))
                }
                for _x in mn_valid_end..mn_range.end {
                    packer.write(T::default())
                }
            }
        }
    }}

    #[inline]
    pub unsafe fn pack_segment<'a, 'b>(
        &self,
        mut pb: impl std::borrow::BorrowMut<TensorView<'a>>,
        b: impl std::borrow::Borrow<TensorView<'b>>,
        k_axis: usize,
        mn_axis: usize,
        k_range: Range<usize>,
        mn_range: Range<usize>,
    ) {
        debug_assert!(pb.borrow().len() >= self.len(k_range.len(), mn_range.len()));
        let pb = pb.borrow_mut();
        let b = b.borrow();
        let dt = pb.datum_type();
        unsafe {
            dispatch_copy!(Self::pack_t(dt)(
                self,
                pb.as_ptr_mut_unchecked(),
                b.as_ptr_unchecked(),
                b.shape()[mn_axis],
                b.strides()[k_axis],
                b.strides()[mn_axis],
                k_range,
                mn_range
            ));
        }
    }

    pub fn write_with_k_outer<'p, T: Copy + Debug>(
        &self,
        pb: *mut T,
        k: usize,
        mn: usize,
    ) -> KOutWriter<'p, T> {
        KOutWriter::new(pb, self.r, self.single_panel_len(k), mn, k)
    }

    pub fn write_single_panel_with_k_outer<'p, T: Copy + Debug>(
        &self,
        pb: *mut T,
    ) -> KOutSinglePanelWriter<'p, T> {
        KOutSinglePanelWriter::new(pb)
    }

    pub fn write_with_k_inner<'p, T: Copy + Debug>(
        &self,
        pb: *mut T,
        k: usize,
        mn: usize,
    ) -> KInWriter<'p, T> {
        let panel_len = self.single_panel_len(k);
        KInWriter::new(pb, panel_len, self.r, mn, k)
    }
}

pub trait PackingWriter<T: Copy> {
    fn write(&mut self, t: T);

    /// Write a contiguous slice of values. The default implementation falls
    /// back to per-element `write`; concrete writers may override with a
    /// `memcpy`-class fast path when the destination layout permits it.
    ///
    /// The output produced by `write_slice(s)` must be byte-identical to
    /// `for &t in s { self.write(t); }` for any input.
    #[inline]
    fn write_slice(&mut self, ts: &[T]) {
        for t in ts {
            self.write(*t);
        }
    }
}

#[derive(Debug)]
pub struct KOutSinglePanelWriter<'p, T>
where
    T: Copy + std::fmt::Debug,
{
    ptr: *mut T,
    _phantom: PhantomData<&'p T>,
}

impl<'p, T> KOutSinglePanelWriter<'p, T>
where
    T: Copy + std::fmt::Debug,
{
    pub fn new(ptr: *mut T) -> KOutSinglePanelWriter<'p, T> {
        KOutSinglePanelWriter { ptr, _phantom: PhantomData }
    }
}

impl<T> PackingWriter<T> for KOutSinglePanelWriter<'_, T>
where
    T: Copy + std::fmt::Debug,
{
    #[inline(always)]
    fn write(&mut self, t: T) {
        unsafe {
            *self.ptr = t;
            self.ptr = self.ptr.offset(1);
        }
    }

    #[inline]
    fn write_slice(&mut self, ts: &[T]) {
        // KOutSinglePanelWriter writes elements consecutively with no panel
        // boundaries. A direct `copy_nonoverlapping` is byte-identical to the
        // per-element loop.
        unsafe {
            std::ptr::copy_nonoverlapping(ts.as_ptr(), self.ptr, ts.len());
            self.ptr = self.ptr.add(ts.len());
        }
    }
}

#[derive(Debug)]
pub struct KOutWriter<'p, T>
where
    T: Copy + std::fmt::Debug,
{
    ptr: *mut T,
    panels: usize,
    panel_width: usize,
    last_panel_width: usize,
    remain: usize,
    current_panel: usize,
    next_panel: isize,
    next_lane: isize,
    _phantom: PhantomData<&'p T>,
}

impl<'p, T> KOutWriter<'p, T>
where
    T: Copy + std::fmt::Debug,
{
    pub fn new(
        ptr: *mut T,
        panel_width: usize,
        panel_len: usize,
        mn: usize,
        _k: usize,
    ) -> KOutWriter<'p, T> {
        let panels = mn.divceil(panel_width);
        let last_panel_width = mn - (panels - 1) * panel_width;
        KOutWriter {
            ptr,
            panels,
            panel_width,
            last_panel_width,
            remain: if panels > 1 { panel_width } else { last_panel_width },
            current_panel: 0,
            next_panel: (panel_len - panel_width) as isize,
            next_lane: (panel_width - last_panel_width) as isize
                - (panel_len * (panels - 1)) as isize,
            _phantom: PhantomData,
        }
    }
}

impl<T> PackingWriter<T> for KOutWriter<'_, T>
where
    T: Copy + std::fmt::Debug,
{
    #[inline(always)]
    fn write(&mut self, t: T) {
        unsafe {
            *self.ptr = t;
            self.remain -= 1;
            self.ptr = self.ptr.offset(1);
            if self.remain == 0 {
                self.current_panel += 1;
                if self.current_panel == self.panels {
                    self.ptr = self.ptr.offset(self.next_lane);
                    self.current_panel = 0;
                } else {
                    self.ptr = self.ptr.offset(self.next_panel);
                }
                if self.current_panel == self.panels - 1 {
                    self.remain = self.last_panel_width;
                } else {
                    self.remain = self.panel_width;
                }
            }
        }
    }

    #[inline]
    fn write_slice(&mut self, ts: &[T]) {
        // Fast path: the slice fits entirely within the current panel. Writes
        // are then guaranteed to be `ts.len()` consecutive memory locations
        // followed by the same panel/lane bookkeeping the per-element path
        // performs. This produces byte-identical output to a per-element loop.
        //
        // When the slice would cross a panel boundary, fall back to the
        // per-element path so all transition logic stays in one place.
        let n = ts.len();
        if n == 0 {
            return;
        }
        if n < self.remain {
            // Strictly inside the current panel: bulk copy, then advance.
            unsafe {
                std::ptr::copy_nonoverlapping(ts.as_ptr(), self.ptr, n);
                self.ptr = self.ptr.add(n);
            }
            self.remain -= n;
        } else if n == self.remain {
            // Exactly fills the current panel: bulk copy, then run the same
            // panel-transition bookkeeping that `write` does on its final
            // element. The transition is performed unconditionally here
            // (rather than calling `write` for the last element) to keep the
            // semantics identical even when the trait is inlined separately.
            unsafe {
                std::ptr::copy_nonoverlapping(ts.as_ptr(), self.ptr, n);
                self.ptr = self.ptr.add(n);
                self.current_panel += 1;
                if self.current_panel == self.panels {
                    self.ptr = self.ptr.offset(self.next_lane);
                    self.current_panel = 0;
                } else {
                    self.ptr = self.ptr.offset(self.next_panel);
                }
                if self.current_panel == self.panels - 1 {
                    self.remain = self.last_panel_width;
                } else {
                    self.remain = self.panel_width;
                }
            }
        } else {
            // Spans a panel boundary. Fall back to per-element writes so the
            // panel-transition state machine handles every step.
            for t in ts {
                self.write(*t);
            }
        }
    }
}

#[derive(Debug)]
pub struct KInWriter<'p, T>
where
    T: Copy + Debug,
{
    ptr: *mut T,
    k: usize,
    panels: usize,
    panel_width: usize,
    last_panel_width: usize,
    remain_on_k: usize,
    remain_on_mn: usize,
    current_panel: usize,
    next_mn_offset: isize,
    next_panel_offset: isize,
    _phantom: PhantomData<&'p T>,
}

impl<'p, T> KInWriter<'p, T>
where
    T: Copy + Debug,
{
    pub fn new(
        ptr: *mut T,
        panel_len: usize,
        panel_width: usize,
        mn: usize,
        k: usize,
    ) -> KInWriter<'p, T> {
        let panels = mn.divceil(panel_width);
        let last_panel_width = mn - (panels - 1) * panel_width;
        KInWriter {
            ptr,
            k,
            panels,
            panel_width,
            last_panel_width,
            remain_on_k: k,
            remain_on_mn: if panels == 1 { last_panel_width } else { panel_width },
            current_panel: 0,
            next_mn_offset: 1 - (k * panel_width) as isize,
            next_panel_offset: panel_len as isize - (k * panel_width + panel_width - 1) as isize,
            //                 ^ next panel     ^    ^ rewind left ^   ^ rewind up   ^
            _phantom: PhantomData,
        }
    }
}

impl<T> PackingWriter<T> for KInWriter<'_, T>
where
    T: Copy + std::fmt::Debug,
{
    #[inline(always)]
    fn write(&mut self, t: T) {
        unsafe {
            *self.ptr = t;
            self.remain_on_k -= 1;
            self.ptr = self.ptr.add(self.panel_width);
            if self.remain_on_k == 0 {
                self.remain_on_k = self.k;
                self.remain_on_mn -= 1;
                if self.remain_on_mn > 0 {
                    self.ptr = self.ptr.offset(self.next_mn_offset);
                } else {
                    self.ptr = self.ptr.offset(self.next_panel_offset);
                    self.current_panel += 1;
                    if self.current_panel == self.panels - 1 {
                        self.remain_on_mn = self.last_panel_width;
                    } else {
                        self.remain_on_mn = self.panel_width;
                    }
                }
            }
        }
    }
}

#[inline(never)]
unsafe fn pack_mn_major<Chunk: Copy>(
    b: *const u8,
    packed: *mut u8,
    panel_len: usize,
    k_stride_bytes: isize,
    mn_range_bytes: Range<usize>,
    k_range: Range<usize>,
) {
    unsafe {
        let mnr = std::mem::size_of::<Chunk>();
        let full_panes = mn_range_bytes.len() / mnr;
        let partial_pane = mn_range_bytes.len() % mnr;
        for k in 0..k_range.len() {
            let mut p_row = packed.add(k * mnr);
            let mut b_row = b.offset(
                (k_range.start + k) as isize * k_stride_bytes + mn_range_bytes.start as isize,
            );
            for _ in 0..full_panes {
                p_row.copy_from_nonoverlapping(b_row, mnr);
                p_row = p_row.add(panel_len);
                b_row = b_row.add(mnr);
            }
            if partial_pane > 0 {
                p_row.copy_from_nonoverlapping(b_row, partial_pane);
            }
        }
    }
}

/// Smallest k-contiguous block (in elements) worth transposing with the armv7
/// NEON tile rather than the scalar tail. Below it the tile's setup does not
/// amortise on armv7's narrow in-order NEON; the crossover sits in the gap
/// between the small activation packs (≤3072) and the large ones (≥5120) that
/// the wake-word models produce, and holds on both cortex-a7 and cortex-a9.
const ARMV7_TILE_MIN_ELEMS: usize = 4096;

/// Whether the 32-bit arm NEON transpose leaves may run: their mnemonics are
/// only valid, and `pack_k_major` only routes to them, when the CPU has NEON.
#[cfg(target_arch = "arm")]
#[inline]
fn armv7_has_neon() -> bool {
    crate::arm32::has_neon()
}

#[cfg(not(target_arch = "arm"))]
#[inline]
fn armv7_has_neon() -> bool {
    false
}

/// Pack a k-contiguous source block: transpose it into the k-inner packed
/// layout, where source element `(mn, k)` of the block lands at
/// `(mn / r) * panel_len + k * r + mn % r`. `b` points at element `(0, 0)`, and
/// `mn_len` counts valid mn columns only: nothing outside the block is read.
///
/// The result must stay byte-identical to feeding [`KInWriter`] mn-outer /
/// k-inner. Stores are strided by `r`, so the block moves as 4x4 tiles, k-outer
/// so that each panel is filled front to back; the tails go element by element.
#[inline(never)]
unsafe fn pack_k_major<T: Copy>(
    b: *const T,
    packed: *mut T,
    panel_len: usize,
    r: usize,
    mn_stride: isize,
    k_len: usize,
    mn_len: usize,
) {
    unsafe {
        // The tile is vectorised on aarch64 (always) and on 32-bit arm only for
        // the 2- and 4-byte NEON leaves, and there only when NEON is present.
        // Any other arm case would spill 16 live tile values through the stack,
        // so it takes the byte-identical scalar tail instead. armv7's weak NEON
        // also loses to the scalar store on small blocks, where the tile setup
        // does not amortise; below ARMV7_TILE_MIN_ELEMS it takes the tail too.
        let tile = if cfg!(target_arch = "arm") {
            armv7_has_neon()
                && matches!(std::mem::size_of::<T>(), 2 | 4)
                && k_len * mn_len >= ARMV7_TILE_MIN_ELEMS
        } else {
            true
        };
        for panel in 0..mn_len.divceil(r) {
            let panel_mn = panel * r;
            let panel_width = r.min(mn_len - panel_mn);
            let src = b.offset(panel_mn as isize * mn_stride);
            let dst = packed.add(panel * panel_len);
            let tiled_mn = if tile { panel_width / 4 * 4 } else { 0 };
            let tiled_k = k_len / 4 * 4;
            for k in (0..tiled_k).step_by(4) {
                for x in (0..tiled_mn).step_by(4) {
                    transpose_4x4(
                        src.offset(x as isize * mn_stride + k as isize),
                        mn_stride,
                        dst.add(k * r + x),
                        r,
                    );
                }
            }
            for k in tiled_k..k_len {
                for x in 0..tiled_mn {
                    *dst.add(k * r + x) = *src.offset(x as isize * mn_stride + k as isize);
                }
            }
            for x in tiled_mn..panel_width {
                let row = src.offset(x as isize * mn_stride);
                for k in 0..k_len {
                    *dst.add(k * r + x) = *row.add(k);
                }
            }
        }
    }
}

/// Transpose a 4x4 tile: `src` rows are `src_stride` apart with contiguous
/// elements, `dst` rows are `dst_stride` apart with contiguous elements. Both
/// strides count elements and may leave the tiles unaligned. Specialised by
/// element width where a vector transpose exists, portable everywhere else.
#[inline(always)]
unsafe fn transpose_4x4<T: Copy>(src: *const T, src_stride: isize, dst: *mut T, dst_stride: usize) {
    unsafe {
        // Alignment is part of the test: a 4-byte T of alignment 2 (Complex<i16>)
        // must not be moved through a lane type it cannot be aligned for.
        #[cfg(target_arch = "aarch64")]
        if std::mem::size_of::<T>() == 4 && std::mem::align_of::<T>() == 4 {
            transpose_4x4_neon_32(src as _, src_stride, dst as _, dst_stride);
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if std::mem::size_of::<T>() == 2 && std::mem::align_of::<T>() == 2 {
            transpose_4x4_neon_16(src as _, src_stride, dst as _, dst_stride);
            return;
        }
        // 32-bit arm: NEON via asm, since both the intrinsics and
        // `#[target_feature(enable = "neon")]` are unstable on this target.
        // Reached only through pack_k_major's tiled path, which on arm runs
        // solely when has_neon() is true, so the NEON these emit is present.
        #[cfg(target_arch = "arm")]
        if std::mem::size_of::<T>() == 4 && std::mem::align_of::<T>() == 4 {
            transpose_4x4_neon_armv7_32(src as _, src_stride, dst as _, dst_stride);
            return;
        }
        #[cfg(target_arch = "arm")]
        if std::mem::size_of::<T>() == 2 && std::mem::align_of::<T>() == 2 {
            transpose_4x4_neon_armv7_16(src as _, src_stride, dst as _, dst_stride);
            return;
        }
        let tile: [[T; 4]; 4] = std::array::from_fn(|i| {
            let row = src.offset(i as isize * src_stride);
            std::array::from_fn(|j| *row.add(j))
        });
        for j in 0..4 {
            let out = dst.add(j * dst_stride);
            for (i, row) in tile.iter().enumerate() {
                *out.add(i) = row[j];
            }
        }
    }
}

/// 4x4 transpose of 32-bit lanes: four `ld1`, eight `trn`, four `st1`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn transpose_4x4_neon_32(
    src: *const u32,
    src_stride: isize,
    dst: *mut u32,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        let a = vld1q_u32(src);
        let b = vld1q_u32(src.offset(src_stride));
        let c = vld1q_u32(src.offset(2 * src_stride));
        let d = vld1q_u32(src.offset(3 * src_stride));
        let ab_even = vreinterpretq_u64_u32(vtrn1q_u32(a, b));
        let ab_odd = vreinterpretq_u64_u32(vtrn2q_u32(a, b));
        let cd_even = vreinterpretq_u64_u32(vtrn1q_u32(c, d));
        let cd_odd = vreinterpretq_u64_u32(vtrn2q_u32(c, d));
        vst1q_u32(dst, vreinterpretq_u32_u64(vtrn1q_u64(ab_even, cd_even)));
        vst1q_u32(dst.add(dst_stride), vreinterpretq_u32_u64(vtrn1q_u64(ab_odd, cd_odd)));
        vst1q_u32(dst.add(2 * dst_stride), vreinterpretq_u32_u64(vtrn2q_u64(ab_even, cd_even)));
        vst1q_u32(dst.add(3 * dst_stride), vreinterpretq_u32_u64(vtrn2q_u64(ab_odd, cd_odd)));
    }
}

/// 4x4 transpose of 16-bit lanes, on 64-bit halves of the vector registers.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn transpose_4x4_neon_16(
    src: *const u16,
    src_stride: isize,
    dst: *mut u16,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        let a = vld1_u16(src);
        let b = vld1_u16(src.offset(src_stride));
        let c = vld1_u16(src.offset(2 * src_stride));
        let d = vld1_u16(src.offset(3 * src_stride));
        let ab_even = vreinterpret_u32_u16(vtrn1_u16(a, b));
        let ab_odd = vreinterpret_u32_u16(vtrn2_u16(a, b));
        let cd_even = vreinterpret_u32_u16(vtrn1_u16(c, d));
        let cd_odd = vreinterpret_u32_u16(vtrn2_u16(c, d));
        vst1_u16(dst, vreinterpret_u16_u32(vtrn1_u32(ab_even, cd_even)));
        vst1_u16(dst.add(dst_stride), vreinterpret_u16_u32(vtrn1_u32(ab_odd, cd_odd)));
        vst1_u16(dst.add(2 * dst_stride), vreinterpret_u16_u32(vtrn2_u32(ab_even, cd_even)));
        vst1_u16(dst.add(3 * dst_stride), vreinterpret_u16_u32(vtrn2_u32(ab_odd, cd_odd)));
    }
}

/// 4x4 transpose of 32-bit lanes on 32-bit arm: four `vld1.32`, two `vtrn.32`,
/// two `vswp`, four `vst1.32`, all in q0-q3. Strides count elements. NEON is
/// enabled locally with `.fpu neon` because it cannot be turned on through
/// `-C target-feature` on this target's stable channel.
///
/// # Safety
/// The CPU must have NEON: there is no `#[target_feature(enable = "neon")]` on
/// this target to assert it (unstable), so the caller guarantees it, which
/// `pack_k_major` does by only tiling under `arm32::has_neon()`.
#[cfg(target_arch = "arm")]
#[inline(always)]
unsafe fn transpose_4x4_neon_armv7_32(
    src: *const u32,
    src_stride: isize,
    dst: *mut u32,
    dst_stride: usize,
) {
    use std::arch::asm;
    let ss = src_stride * 4;
    let ds = (dst_stride * 4) as isize;
    let src = src as *const u8;
    let dst = dst as *mut u8;
    unsafe {
        asm!(
            ".fpu neon",
            "vld1.32 {{d0, d1}}, [{s0}]",
            "vld1.32 {{d2, d3}}, [{s1}]",
            "vld1.32 {{d4, d5}}, [{s2}]",
            "vld1.32 {{d6, d7}}, [{s3}]",
            "vtrn.32 q0, q1",
            "vtrn.32 q2, q3",
            "vswp d1, d4",
            "vswp d3, d6",
            "vst1.32 {{d0, d1}}, [{o0}]",
            "vst1.32 {{d2, d3}}, [{o1}]",
            "vst1.32 {{d4, d5}}, [{o2}]",
            "vst1.32 {{d6, d7}}, [{o3}]",
            s0 = in(reg) src,
            s1 = in(reg) src.offset(ss),
            s2 = in(reg) src.offset(2 * ss),
            s3 = in(reg) src.offset(3 * ss),
            o0 = in(reg) dst,
            o1 = in(reg) dst.offset(ds),
            o2 = in(reg) dst.offset(2 * ds),
            o3 = in(reg) dst.offset(3 * ds),
            out("q0") _,
            out("q1") _,
            out("q2") _,
            out("q3") _,
            options(nostack),
        );
    }
}

/// 4x4 transpose of 16-bit lanes on 32-bit arm, on 64-bit d registers: four
/// `vld1.16`, two `vtrn.16`, two `vtrn.32`, four `vst1.16`. Same NEON and
/// safety contract as [`transpose_4x4_neon_armv7_32`].
#[cfg(target_arch = "arm")]
#[inline(always)]
unsafe fn transpose_4x4_neon_armv7_16(
    src: *const u16,
    src_stride: isize,
    dst: *mut u16,
    dst_stride: usize,
) {
    use std::arch::asm;
    let ss = src_stride * 2;
    let ds = (dst_stride * 2) as isize;
    let src = src as *const u8;
    let dst = dst as *mut u8;
    unsafe {
        asm!(
            ".fpu neon",
            "vld1.16 {{d0}}, [{s0}]",
            "vld1.16 {{d1}}, [{s1}]",
            "vld1.16 {{d2}}, [{s2}]",
            "vld1.16 {{d3}}, [{s3}]",
            "vtrn.16 d0, d1",
            "vtrn.16 d2, d3",
            "vtrn.32 d0, d2",
            "vtrn.32 d1, d3",
            "vst1.16 {{d0}}, [{o0}]",
            "vst1.16 {{d1}}, [{o1}]",
            "vst1.16 {{d2}}, [{o2}]",
            "vst1.16 {{d3}}, [{o3}]",
            s0 = in(reg) src,
            s1 = in(reg) src.offset(ss),
            s2 = in(reg) src.offset(2 * ss),
            s3 = in(reg) src.offset(3 * ss),
            o0 = in(reg) dst,
            o1 = in(reg) dst.offset(ds),
            o2 = in(reg) dst.offset(2 * ds),
            o3 = in(reg) dst.offset(3 * ds),
            out("d0") _,
            out("d1") _,
            out("d2") _,
            out("d3") _,
            options(nostack),
        );
    }
}

// K=4-inner packing writer (PackedI8K4 layout), fed in K-OUTER order (same feed
// as KOutWriter, used by the im2col patchers): for each k, all mn. Within a panel,
// element (k, local_mn) lands at (k/4)*r*4 + local_mn*4 + (k%4), so consecutive mn
// for a fixed k are stride-4 stores.
#[derive(Debug)]
pub struct KOut4Writer<'p, T>
where
    T: Copy + std::fmt::Debug,
{
    base: *mut T,
    r4: usize,        // r * 4
    panel_len: usize, // k_aligned * r
    panels: usize,
    panel_width: usize,
    last_panel_width: usize,
    kb: usize, // k / 4
    kr: usize, // k % 4
    panel: usize,
    local_mn: usize,
    _phantom: PhantomData<&'p T>,
}

impl<'p, T> KOut4Writer<'p, T>
where
    T: Copy + std::fmt::Debug,
{
    pub fn new(base: *mut T, r: usize, panel_len: usize, mn: usize) -> KOut4Writer<'p, T> {
        let panels = mn.divceil(r).max(1);
        let last_panel_width = mn - (panels - 1) * r;
        KOut4Writer {
            base,
            r4: r * 4,
            panel_len,
            panels,
            panel_width: r,
            last_panel_width,
            kb: 0,
            kr: 0,
            panel: 0,
            local_mn: 0,
            _phantom: PhantomData,
        }
    }
    #[inline(always)]
    fn panel_width(&self) -> usize {
        if self.panel == self.panels - 1 { self.last_panel_width } else { self.panel_width }
    }
    #[inline(always)]
    fn advance(&mut self, by: usize) {
        self.local_mn += by;
        if self.local_mn >= self.panel_width() {
            self.local_mn = 0;
            self.panel += 1;
            if self.panel == self.panels {
                self.panel = 0;
                self.kr += 1;
                if self.kr == 4 {
                    self.kr = 0;
                    self.kb += 1;
                }
            }
        }
    }
}

impl<T> PackingWriter<T> for KOut4Writer<'_, T>
where
    T: Copy + std::fmt::Debug,
{
    #[inline(always)]
    fn write(&mut self, t: T) {
        unsafe {
            let off = self.panel * self.panel_len + self.kb * self.r4 + self.local_mn * 4 + self.kr;
            *self.base.add(off) = t;
        }
        self.advance(1);
    }

    #[inline]
    fn write_slice(&mut self, ts: &[T]) {
        let n = ts.len();
        if n == 0 {
            return;
        }
        let pw = self.panel_width();
        if self.local_mn + n <= pw {
            // Whole slice stays inside the current (panel, k): tight stride-4 store.
            unsafe {
                let mut d = self.base.add(
                    self.panel * self.panel_len + self.kb * self.r4 + self.local_mn * 4 + self.kr,
                );
                for &t in ts {
                    *d = t;
                    d = d.add(4);
                }
            }
            self.advance(n);
        } else {
            for &t in ts {
                self.write(t);
            }
        }
    }
}

// K=4-inner packing for SDOT/relaxed-dot int8 matmul: 4 contiguous K per mn-lane.
// Layout: out[(k/4)*r*4 + m*4 + (k%4)] = src[m,k]. k_alignment=4. Matmul path uses
// pack_view; the conv im2col patchers feed write_with_k_outer in K-outer order.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PackedI8K4 {
    pub r: usize,
    pub align: usize,
}
impl PackedI8K4 {
    pub fn new(r: usize) -> Self {
        PackedI8K4 { r, align: 16 }
    }
    fn panel(&self, k: usize) -> usize {
        (k.div_ceil(4) * 4) * self.r
    }
    pub fn single_panel_len(&self, k: usize) -> usize {
        self.panel(k)
    }
    pub fn len(&self, k: usize, mn: usize) -> usize {
        mn.divceil(self.r) * self.panel(k)
    }
    pub fn alignment(&self) -> usize {
        self.align
    }
    // One-pass K-outer writer for the conv im2col patchers (fed: for each k, all mn).
    pub fn write_with_k_outer<'p, T: Copy + std::fmt::Debug>(
        &self,
        pb: *mut T,
        k: usize,
        mn: usize,
    ) -> KOut4Writer<'p, T> {
        KOut4Writer::new(pb, self.r, self.panel(k), mn)
    }
    // K=4-inner pack from a (possibly strided) view: out[(k/4)*r*4 + m*4 + (k%4)] = src[m,k].
    pub fn pack_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        let k = t.shape()[k_axis];
        let mn = t.shape()[mn_axis];
        let kp = k.div_ceil(4) * 4;
        let pl = kp * self.r;
        let panels = mn.div_ceil(self.r);
        let st = t.strides();
        let mut blob = unsafe { Blob::new_for_size_and_align(panels * pl, self.align) };
        blob.as_bytes_mut().fill(0);
        let (ks, ms) = (st[k_axis], st[mn_axis]);
        let kblocks = kp / 4;
        unsafe {
            let src = t.as_ptr_unchecked::<i8>();
            let dst = blob.as_mut_ptr() as *mut i8;
            for p in 0..panels {
                let pw = self.r.min(mn - p * self.r);
                let panel = dst.add(p * pl);
                let mn0 = (p * self.r) as isize;
                for kb in 0..kblocks {
                    for kr in 0..4 {
                        let kk = kb * 4 + kr;
                        if kk >= k {
                            break;
                        }
                        let srow = src.offset(kk as isize * ks + mn0 * ms);
                        let dcol = panel.add(kb * self.r * 4 + kr);
                        for lm in 0..pw {
                            *dcol.add(lm * 4) = *srow.offset(lm as isize * ms);
                        }
                    }
                }
            }
        }
        Ok(Box::new(EagerPackedInput {
            fact: PackedExoticFact { format: Box::new(self.clone()), mn: mn.to_dim(), k },
            packed: blob.into(),
            panel_bytes: pl,
            mn,
        }))
    }
}
impl std::fmt::Display for PackedI8K4 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "I8K4[{}]", self.r)
    }
}
impl MMMInputFormat for PackedI8K4 {
    fn prepare_tensor(&self, t: &Tensor, k_axis: usize, mn_axis: usize) -> TractResult<Tensor> {
        Ok(PackedMatrixStorage::new(self.prepare_one(t, k_axis, mn_axis)?)
            .into_tensor(t.datum_type()))
    }
    fn prepare_one_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        self.pack_view(t, k_axis, mn_axis)
    }
    fn precursor(&self) -> WeightType {
        WeightType::Plain(i8::datum_type())
    }
    fn r(&self) -> usize {
        self.r
    }
    fn k_alignment(&self) -> usize {
        4
    }
    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        o: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        o.downcast_ref::<PackedI8K4>().filter(|x| x.r == self.r).map(|_| self as _)
    }
    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        mn.divceil(self.r) * self.panel(k.to_usize().unwrap_or(0))
    }
    fn extract_at_mn_f16(&self, _: &EagerPackedInput, _: usize, _: &mut [f16]) -> TractResult<()> {
        bail!("no f16 extract")
    }
    fn extract_at_mn_f32(&self, _: &EagerPackedInput, _: usize, _: &mut [f32]) -> TractResult<()> {
        bail!("no f32 extract")
    }
}

pub trait Packing {
    fn packing(r: usize) -> PackedFormat;
}

impl<D: Datum> Packing for D {
    fn packing(r: usize) -> PackedFormat {
        PackedFormat::new(Self::datum_type(), r, vector_size())
    }
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use proptest::prelude::*;
    use tract_data::internal::num_integer::Integer;
    use tract_data::internal::tract_ndarray::Zip;
    use tract_data::internal::*;
    use tract_ndarray::prelude::*;

    #[derive(Debug)]
    struct PackProblem {
        k: usize,
        mn: usize,
        is_a: bool,
        r: usize,
        k_range: Range<usize>,
        mn_range: Range<usize>,
        align_panel: usize,
    }

    impl PackProblem {
        fn input(&self) -> Array2<u32> {
            let shape = if self.is_a { (self.mn, self.k) } else { (self.k, self.mn) };
            let data = (0..(self.k * self.mn) as u32).collect();
            Array2::from_shape_vec(shape, data).unwrap()
        }

        fn packer(&self) -> Array2<u32> {
            let panels = self.mn_range.len().divceil(self.r);
            let packer = super::PackedFormat::new(u32::datum_type(), self.r, self.align_panel)
                .with_end_padding_record(0);
            let input = self.input().into_tensor();
            let panel_len = packer.single_panel_len(self.k_range.len());
            let mut output =
                Tensor::zero::<u32>(&[packer.len(self.k_range.len(), self.mn_range.len())])
                    .unwrap();
            unsafe {
                packer.pack_segment(
                    output.view_mut(),
                    input.view(),
                    self.is_a as usize,
                    !self.is_a as usize,
                    self.k_range.clone(),
                    self.mn_range.clone(),
                )
            };
            output
                .into_plain_array::<u32>()
                .unwrap()
                .into_shape_with_order((panels, panel_len))
                .unwrap()
        }

        fn reference(&self) -> Array2<u32> {
            let input = self.input();
            let panels = self.mn_range.len().divceil(self.r);
            let len = Integer::next_multiple_of(&(self.k_range.len() * self.r), &self.align_panel);
            Array2::from_shape_fn([panels, len], |(panel, z)| {
                let k = z / self.r;
                let x = z % self.r;
                let mn = panel * self.r + x + self.mn_range.start;
                let k = k + self.k_range.start;
                let coords = if self.is_a { (mn, k) } else { (k, mn) };
                *input.get(coords).unwrap_or(&0)
            })
        }

        fn valid(&self) -> Array2<bool> {
            let panels = self.mn_range.len().divceil(self.r);
            let len = Integer::next_multiple_of(&(self.k_range.len() * self.r), &self.align_panel);
            Array2::from_shape_fn([panels, len], |(panel, z)| {
                let k = z / self.r;
                let x = z % self.r;
                let k = k + self.k_range.start;
                let mn = panel * self.r + x + self.mn_range.start;
                k < self.k_range.end.min(self.k) && mn < self.mn_range.end.min(self.mn)
            })
        }

        fn check(&self) {
            let mut packer = self.packer();
            let mut reference = self.reference();
            let valid = self.valid();
            Zip::from(&mut packer).and(&valid).for_each(|p, v| *p = if *v { *p } else { -1 as _ });
            Zip::from(&mut reference)
                .and(&valid)
                .for_each(|p, v| *p = if *v { *p } else { -1 as _ });
            assert_eq!(packer, reference);
        }
    }

    impl Arbitrary for PackProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<PackProblem>;
        fn arbitrary_with(_args: ()) -> Self::Strategy {
            (any::<bool>(), 1usize..9, 1usize..20, 1usize..20)
                .prop_flat_map(|(is_a, r, k, mn)| {
                    (
                        Just((is_a, r, k, mn)),
                        sub_range_strat(0..k),
                        sub_range_strat(0..mn),
                        1usize..5,
                    )
                })
                .prop_map(|((is_a, r, k, mn), k_range, mn_range, align_panel)| PackProblem {
                    k,
                    mn,
                    is_a,
                    r,
                    k_range,
                    mn_range,
                    align_panel,
                })
                .boxed()
        }
    }

    fn sub_range_strat(range: Range<usize>) -> BoxedStrategy<Range<usize>> {
        (0..range.len())
            .prop_flat_map(|cropped| (Just(cropped), 0..=cropped))
            .prop_map(move |(cropped, left)| range.start + left..range.end - (cropped - left))
            .boxed()
    }

    proptest::proptest! {
        #[test]
        fn prop(pb in any::<PackProblem>()) {
            pb.check();
        }

        #[test]
        fn subrange_prop(_range in sub_range_strat(0..20)) {
        }

    }

    // ---- k-contiguous packing -----------------------------------------------
    //
    // A source whose k axis is contiguous (the shape an activation arrives in)
    // is packed with a blocked transpose, and the tiles are SIMD on some
    // targets. The result must stay byte-identical to feeding `KInWriter`
    // element by element, for every element width and for every panel width —
    // including the ones a 4x4 tile does not divide.
    #[derive(Debug, Clone)]
    struct PackKMajorProblem {
        k: usize,
        mn: usize,
        r: usize,
        align_panel: usize,
        k_range: Range<usize>,
        mn_range: Range<usize>,
    }

    impl PackKMajorProblem {
        fn check<T: Datum + Copy + num_traits::Zero>(&self, value: impl Fn(usize, usize) -> T) {
            let input =
                Array2::from_shape_fn((self.mn, self.k), |(x, k)| value(x, k)).into_tensor();
            let packer = super::PackedFormat::new(T::datum_type(), self.r, self.align_panel);
            let len = packer.len(self.k_range.len(), self.mn_range.len());

            let mut packed = Tensor::zero::<T>(&[len]).unwrap();
            unsafe {
                // [mn, k]: k_axis 1, mn_axis 0, so k_stride is 1.
                packer.pack_segment(
                    packed.view_mut(),
                    input.view(),
                    1,
                    0,
                    self.k_range.clone(),
                    self.mn_range.clone(),
                )
            };

            let mut reference = Tensor::zero::<T>(&[len]).unwrap();
            let input = input.to_plain_array_view::<T>().unwrap();
            unsafe {
                let mut writer = packer.write_with_k_inner(
                    reference.as_ptr_mut_unchecked::<T>(),
                    self.k_range.len(),
                    self.mn,
                );
                for x in self.mn_range.start..self.mn_range.end.min(self.mn) {
                    for k in self.k_range.clone() {
                        super::PackingWriter::write(&mut writer, input[[x, k]]);
                    }
                }
            }

            assert_eq!(packed, reference, "{self:?} for {:?}", T::datum_type());
        }

        fn check_all_widths(&self) {
            self.check(|x, k| (x * 41 + k * 7) as u32);
            self.check(|x, k| f16::from_f32((x * 41 + k * 7) as f32));
            self.check(|x, k| (x * 41 + k * 7) as u8);
        }
    }

    impl Arbitrary for PackKMajorProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<PackKMajorProblem>;
        fn arbitrary_with(_: ()) -> Self::Strategy {
            // r covers the panel widths of the real f32/f16 kernels plus the
            // ones smaller than a tile.
            (
                prop::sample::select(vec![1usize, 2, 3, 4, 5, 8, 12, 16, 24, 32]),
                1usize..40,
                1usize..40,
            )
                .prop_flat_map(|(r, k, mn)| {
                    (Just((r, k, mn)), 1usize..5, sub_range_strat(0..k), sub_range_strat(0..mn))
                })
                .prop_map(|((r, k, mn), align_panel, k_range, mn_range)| PackKMajorProblem {
                    k,
                    mn,
                    r,
                    align_panel,
                    k_range,
                    mn_range,
                })
                .boxed()
        }
    }

    proptest::proptest! {
        #[test]
        fn pack_k_major_prop(pb in any::<PackKMajorProblem>()) {
            pb.check_all_widths();
        }
    }

    fn k_major(k: usize, mn: usize, r: usize) -> PackKMajorProblem {
        PackKMajorProblem { k, mn, r, align_panel: 1, k_range: 0..k, mn_range: 0..mn }
    }

    #[test]
    fn k_major_exact_tiles() {
        k_major(4, 4, 4).check_all_widths();
        k_major(768, 256, 8).check_all_widths();
        k_major(16, 32, 16).check_all_widths();
    }

    #[test]
    fn k_major_tails() {
        // k % 4, panel width % 4, and both at once.
        for k in [1, 2, 3, 5, 7] {
            k_major(k, 8, 8).check_all_widths();
            k_major(k, 7, 8).check_all_widths();
        }
        k_major(9, 6, 12).check_all_widths();
        k_major(9, 30, 12).check_all_widths();
    }

    #[test]
    fn k_major_tile_over_threshold() {
        // Blocks past the armv7 size gate, so the NEON tile runs (not just the
        // scalar tail the proptest's small blocks take there) while still
        // hitting the k, panel-width, and narrow-last-panel tails.
        k_major(129, 41, 8).check_all_widths();
        k_major(160, 26, 16).check_all_widths();
        k_major(130, 44, 12).check_all_widths();
        k_major(140, 30, 8).check_all_widths();
    }

    #[test]
    fn k_major_narrower_than_a_tile() {
        for r in [1, 2, 3] {
            k_major(9, 7, r).check_all_widths();
        }
    }

    #[test]
    fn k_major_segments() {
        // A cropped k_range must still land at panel offset 0.
        PackKMajorProblem { k: 20, mn: 20, r: 8, align_panel: 1, k_range: 3..17, mn_range: 0..20 }
            .check_all_widths();
        PackKMajorProblem { k: 20, mn: 20, r: 8, align_panel: 1, k_range: 0..20, mn_range: 4..12 }
            .check_all_widths();
        // mn_range reaching past mn: the invalid columns are left untouched.
        PackKMajorProblem { k: 20, mn: 20, r: 8, align_panel: 1, k_range: 0..20, mn_range: 16..24 }
            .check_all_widths();
    }

    // ---- PackedI8K4 (K=4-inner SMOPA/SDOT layout) dedicated tests ----------
    //
    // PackedI8K4 has two independent producers that MUST agree byte-for-byte:
    //   * `pack_view`           — the matmul path, reads a (possibly strided)
    //                             TensorView and packs in one shot.
    //   * `write_with_k_outer`  — the conv/im2col path, fed element-by-element
    //                             in K-OUTER order (for each k, all mn).
    // Both must equal the canonical layout
    //     out[panel*pl + (k/4)*r*4 + local_mn*4 + (k%4)] = src[k, panel*r+local_mn]
    // with pl = ceil(K/4)*4 * r, and every padding byte (K%4 tail, partial last
    // mn panel) left at zero.
    #[derive(Debug, Clone)]
    struct PackI8K4Problem {
        k: usize,
        mn: usize,
        r: usize,
        // false: input tensor is [k, mn] (k_axis=0, mn_axis=1) — contiguous read.
        // true : input tensor is [mn, k] (k_axis=1, mn_axis=0) — strided read,
        //        mirroring how the "A" operand is fed.
        is_a: bool,
    }

    impl PackI8K4Problem {
        // Canonical logical matrix, always indexed [k, mn].
        fn logical(&self) -> Array2<i8> {
            Array2::from_shape_fn((self.k, self.mn), |(kk, m)| {
                (kk.wrapping_mul(31).wrapping_add(m.wrapping_mul(17)).wrapping_add(1)) as i8
            })
        }

        fn panel_len(&self) -> usize {
            (self.k.div_ceil(4) * 4) * self.r
        }

        // The layout every producer must reproduce.
        fn reference(&self) -> Vec<i8> {
            let logical = self.logical();
            let r = self.r;
            let pl = self.panel_len();
            let panels = self.mn.div_ceil(r);
            let mut out = vec![0i8; panels * pl];
            for p in 0..panels {
                let pw = r.min(self.mn - p * r);
                for kk in 0..self.k {
                    for lm in 0..pw {
                        let m = p * r + lm;
                        let off = p * pl + (kk / 4) * r * 4 + lm * 4 + (kk % 4);
                        out[off] = logical[[kk, m]];
                    }
                }
            }
            out
        }

        // The matmul path: pack a TensorView, then read it back panel by panel.
        fn pack_view_bytes(&self) -> Vec<i8> {
            let logical = self.logical();
            let packer = super::PackedI8K4::new(self.r);
            let (tensor, k_axis, mn_axis) = if self.is_a {
                // [mn, k] with entry [m, kk] == logical[kk, m]; reads are strided.
                let a = Array2::from_shape_fn((self.mn, self.k), |(m, kk)| logical[[kk, m]]);
                (a.into_tensor(), 1usize, 0usize)
            } else {
                (logical.clone().into_tensor(), 0usize, 1usize)
            };
            let packed = packer.pack_view(&tensor.view(), k_axis, mn_axis).unwrap();
            let pl = self.panel_len();
            let panels = self.mn.div_ceil(self.r);
            assert_eq!(packed.panels_count(), panels);
            assert_eq!(packed.k(), self.k);
            assert_eq!(packed.mn(), self.mn);
            let mut out = vec![0i8; panels * pl];
            unsafe {
                for p in 0..panels {
                    let ptr = packed.panel_bytes(p, None).unwrap() as *const i8;
                    std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr().add(p * pl), pl);
                }
            }
            out
        }

        // The conv path: feed the writer in K-outer order (for each k, all mn).
        fn writer_bytes(&self) -> Vec<i8> {
            let logical = self.logical();
            let packer = super::PackedI8K4::new(self.r);
            let total = packer.len(self.k, self.mn);
            assert_eq!(total, self.mn.div_ceil(self.r) * self.panel_len());
            let mut buf = vec![0i8; total];
            {
                let mut w = packer.write_with_k_outer(buf.as_mut_ptr(), self.k, self.mn);
                for kk in 0..self.k {
                    for m in 0..self.mn {
                        super::PackingWriter::write(&mut w, logical[[kk, m]]);
                    }
                }
            }
            buf
        }

        fn check(&self) {
            let reference = self.reference();
            assert_eq!(
                self.pack_view_bytes(),
                reference,
                "pack_view disagrees with reference for {self:?}"
            );
            assert_eq!(
                self.writer_bytes(),
                reference,
                "write_with_k_outer disagrees with reference for {self:?}"
            );
        }
    }

    impl Arbitrary for PackI8K4Problem {
        type Parameters = ();
        type Strategy = BoxedStrategy<PackI8K4Problem>;
        fn arbitrary_with(_: ()) -> Self::Strategy {
            // r is the tile width used by the int8 kernels (SMOPA 32, SDOT 8, ...).
            (any::<bool>(), prop::sample::select(vec![4usize, 8, 16, 32]), 1usize..40, 1usize..40)
                .prop_map(|(is_a, r, k, mn)| PackI8K4Problem { k, mn, r, is_a })
                .boxed()
        }
    }

    proptest::proptest! {
        #[test]
        fn pack_i8k4_prop(pb in any::<PackI8K4Problem>()) {
            pb.check();
        }
    }

    fn k4(k: usize, mn: usize, r: usize, is_a: bool) -> PackI8K4Problem {
        PackI8K4Problem { k, mn, r, is_a }
    }

    #[test]
    fn i8k4_smallest() {
        k4(1, 1, 4, false).check();
        k4(1, 1, 4, true).check();
    }

    #[test]
    fn i8k4_exact_tile() {
        // K and mn land exactly on the 4 / r boundaries: no padding anywhere.
        k4(4, 4, 4, false).check();
        k4(8, 32, 32, false).check();
        k4(8, 32, 32, true).check();
    }

    #[test]
    fn i8k4_k_not_multiple_of_4() {
        // K%4 tail must be zero-padded inside each panel.
        for k in [1, 2, 3, 5, 6, 7, 9] {
            k4(k, 4, 4, false).check();
            k4(k, 7, 8, true).check();
        }
    }

    #[test]
    fn i8k4_partial_last_panel() {
        // mn not a multiple of r: last panel is narrower, tail lanes are zero.
        k4(5, 7, 4, false).check();
        k4(5, 7, 4, true).check();
        k4(4, 33, 32, false).check();
        k4(4, 33, 32, true).check();
        k4(3, 1, 32, false).check();
    }

    #[test]
    fn i8k4_single_wide_tile() {
        // One narrow panel inside a wide (r=32) tile.
        k4(7, 1, 32, false).check();
        k4(7, 5, 16, true).check();
    }

    #[test]
    fn i8k4_many_panels() {
        k4(13, 100, 8, false).check();
        k4(13, 100, 8, true).check();
        k4(17, 65, 16, false).check();
    }

    #[test]
    fn simple_b_1() {
        PackProblem {
            k: 2,
            mn: 1,
            is_a: false,
            r: 1,
            k_range: 0..2,
            mn_range: 0..1,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn simple_b_2() {
        PackProblem {
            k: 2,
            mn: 2,
            is_a: false,
            r: 1,
            k_range: 0..2,
            mn_range: 0..2,
            align_panel: 1,
        }
        .check()
    }

    #[test]
    fn simple_b_3() {
        PackProblem {
            k: 2,
            mn: 1,
            is_a: false,
            r: 4,
            k_range: 0..2,
            mn_range: 0..1,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn simple_b_4() {
        PackProblem {
            k: 1,
            mn: 3,
            is_a: false,
            r: 2,
            k_range: 0..1,
            mn_range: 0..3,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn simple_a_1() {
        PackProblem {
            k: 2,
            mn: 2,
            is_a: true,
            r: 1,
            k_range: 0..2,
            mn_range: 0..2,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn simple_a_2() {
        PackProblem {
            k: 2,
            mn: 3,
            is_a: true,
            r: 2,
            k_range: 0..2,
            mn_range: 0..3,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn range_k_0() {
        PackProblem {
            k: 2,
            mn: 1,
            is_a: false,
            r: 1,
            k_range: 1..2,
            mn_range: 0..1,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn range_k_1() {
        PackProblem {
            k: 2,
            mn: 2,
            is_a: false,
            r: 1,
            k_range: 0..2,
            mn_range: 0..1,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn range_k_2() {
        PackProblem {
            k: 2,
            mn: 1,
            is_a: false,
            r: 6,
            k_range: 1..2,
            mn_range: 0..1,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn range_mn_0() {
        PackProblem {
            k: 1,
            mn: 2,
            is_a: false,
            r: 2,
            k_range: 0..1,
            mn_range: 0..1,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn range_b_4() {
        PackProblem {
            k: 1,
            mn: 2,
            is_a: false,
            r: 6,
            k_range: 0..1,
            mn_range: 1..2,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn range_b_5() {
        PackProblem {
            k: 1,
            mn: 7,
            is_a: false,
            r: 6,
            k_range: 0..1,
            mn_range: 1..7,
            align_panel: 1,
        }
        .check();
    }

    #[test]
    fn align_a_1() {
        PackProblem {
            k: 2,
            mn: 2,
            is_a: true,
            r: 1,
            k_range: 0..1,
            mn_range: 0..2,
            align_panel: 2,
        }
        .check();
    }

    #[test]
    fn align_b_1() {
        PackProblem {
            k: 1,
            mn: 1,
            is_a: false,
            r: 1,
            k_range: 0..1,
            mn_range: 0..1,
            align_panel: 2,
        }
        .check();
    }

    #[test]
    fn align_b_2() {
        PackProblem {
            k: 3,
            mn: 1,
            is_a: false,
            r: 1,
            k_range: 0..3,
            mn_range: 0..1,
            align_panel: 2,
        }
        .check();
    }

    #[test]
    fn align_b_3() {
        PackProblem {
            k: 1,
            mn: 1,
            is_a: false,
            r: 3,
            k_range: 0..1,
            mn_range: 0..1,
            align_panel: 2,
        }
        .check();
    }

    #[test]
    fn align_b_4() {
        PackProblem {
            k: 2,
            mn: 1,
            is_a: false,
            r: 1,
            k_range: 0..1,
            mn_range: 0..1,
            align_panel: 2,
        }
        .check();
    }

    #[test]
    fn align_b_5() {
        PackProblem {
            k: 1,
            mn: 5,
            is_a: false,
            r: 4,
            k_range: 0..1,
            mn_range: 0..5,
            align_panel: 3,
        }
        .check();
    }
}
