use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use tract_data::internal::*;

use crate::mmm::{EagerPackedInput, MMMInputFormat, MMMInputValue, PackedOpaqueFact};

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
        Ok(tensor0(Opaque(Arc::new(packed))))
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

    fn same_as(&self, other: &dyn MMMInputFormat) -> bool {
        other.downcast_ref::<Self>().is_some_and(|other| self == other)
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
        ensure!(data.format().same_as(self));
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
        ensure!(data.format().same_as(self));
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
        ensure!(
            t.datum_type().unquantized() == self.dt.unquantized(),
            "Attempting to pack for {self} tensor {t:?}"
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
                fact: PackedOpaqueFact { format: Box::new(self.clone()), mn: mn.to_dim(), k },
                packed: packed.into(),
                panel_bytes,
                mn,
            }))
        }
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
                fact: PackedOpaqueFact { format: Box::new(self.clone()), mn: mn.to_dim(), k },
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
            let mut packer = self.write_with_k_inner(pb, k_range.len(), mn);
            let mn_valid_end = mn_range.end.min(mn);
            for x in mn_range.start..mn_valid_end {
                for k in k_range.clone() {
                    packer.write(*b.offset(x as isize * mn_stride + k as isize))
                }
            }
            // just ignore invalid mn_range
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
            output.into_array::<u32>().unwrap().into_shape_with_order((panels, panel_len)).unwrap()
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
