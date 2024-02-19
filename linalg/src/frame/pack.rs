use std::alloc::Layout;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use tract_data::internal::*;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Packer {
    pub r: usize,
    alignment: usize,
    end_padding_record: usize,
}

impl Packer {
    pub fn new(nr: usize, alignment: usize, end_padding_record: usize) -> Packer {
        Packer { r: nr, alignment, end_padding_record }
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
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
        assert!(k > 0);
        Layout::from_size_align(self.single_panel_len(k) * item_size, self.alignment()).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn pack_t<T: Datum + Copy>(
        &self,
        pb: *mut T,
        b: *const T,
        mn: usize,
        k_stride: isize,
        mn_stride: isize,
        k_range: Range<usize>,
        mn_range: Range<usize>,
    ) {
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
            match rbytes {
                16 => pack_mn_major::<[u8; 16]>(bb, pbb, k_stride_bytes, mn_range_bytes, k_range),
                24 => pack_mn_major::<[u8; 24]>(bb, pbb, k_stride_bytes, mn_range_bytes, k_range),
                32 => pack_mn_major::<[u8; 32]>(bb, pbb, k_stride_bytes, mn_range_bytes, k_range),
                48 => pack_mn_major::<[u8; 48]>(bb, pbb, k_stride_bytes, mn_range_bytes, k_range),
                64 => pack_mn_major::<[u8; 64]>(bb, pbb, k_stride_bytes, mn_range_bytes, k_range),
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
    }

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

    pub unsafe fn pack<'a, 'b>(
        &self,
        pb: impl std::borrow::BorrowMut<TensorView<'a>>,
        b: impl std::borrow::Borrow<TensorView<'b>>,
        k_axis: usize,
        mn_axis: usize,
    ) {
        let k = b.borrow().shape()[k_axis];
        let mn = b.borrow().shape()[mn_axis];
        self.pack_segment(pb, b, k_axis, mn_axis, 0..k, 0..mn);
    }

    pub fn write_with_k_outer<'p, T: Copy + Debug>(
        &self,
        pb: *mut T,
        k: usize,
        mn: usize,
    ) -> KOutWriter<'p, T> {
        KOutWriter::new(pb, self.r, mn, k)
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
        KInWriter::new(pb, self.r, mn, k)
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

impl<'p, T> PackingWriter<T> for KOutSinglePanelWriter<'p, T>
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
    pub fn new(ptr: *mut T, panel_width: usize, mn: usize, k: usize) -> KOutWriter<'p, T> {
        let panels = (mn + panel_width - 1) / panel_width;
        let last_panel_width = mn - (panels - 1) * panel_width;
        KOutWriter {
            ptr,
            panels,
            panel_width,
            last_panel_width,
            remain: if panels > 1 { panel_width } else { last_panel_width },
            current_panel: 0,
            next_panel: ((k - 1) * panel_width) as isize,
            next_lane: panel_width as isize
                - ((last_panel_width + (panels - 1) * panel_width * k) as isize),
            _phantom: PhantomData,
        }
    }
}

impl<'p, T> PackingWriter<T> for KOutWriter<'p, T>
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
    pub fn new(ptr: *mut T, panel_width: usize, mn: usize, k: usize) -> KInWriter<'p, T> {
        let panels = (mn + panel_width - 1) / panel_width;
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
            next_panel_offset: 1 - panel_width as isize,
            _phantom: PhantomData,
        }
    }
}

impl<'p, T> PackingWriter<T> for KInWriter<'p, T>
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
    k_stride_bytes: isize,
    mn_range_bytes: Range<usize>,
    k_range: Range<usize>,
) {
    let mnr = std::mem::size_of::<Chunk>();
    let full_panes = mn_range_bytes.len() / mnr;
    let partial_pane = mn_range_bytes.len() % mnr;
    for k in 0..k_range.len() {
        let mut p_row = packed.add(k * mnr);
        let mut b_row =
            b.offset((k_range.start + k) as isize * k_stride_bytes + mn_range_bytes.start as isize);
        for _ in 0..full_panes {
            p_row.copy_from_nonoverlapping(b_row, mnr);
            p_row = p_row.add(k_range.len() * mnr);
            b_row = b_row.add(mnr);
        }
        if partial_pane > 0 {
            p_row.copy_from_nonoverlapping(b_row, partial_pane);
        }
    }
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use proptest::prelude::*;
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
    }

    impl PackProblem {
        fn input(&self) -> Array2<u32> {
            let shape = if self.is_a { (self.mn, self.k) } else { (self.k, self.mn) };
            let data = (0..(self.k * self.mn) as u32).collect();
            Array2::from_shape_vec(shape, data).unwrap()
        }

        fn packer(&self) -> Array3<u32> {
            let panels = self.mn_range.len().divceil(self.r);
            let packer = super::Packer::new(self.r, 1, 0);
            let input = self.input().into_tensor();
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
                .into_array::<u32>()
                .unwrap()
                .into_shape((panels, self.k_range.len(), self.r))
                .unwrap()
        }

        fn reference(&self) -> Array3<u32> {
            let input = self.input();
            let panels = self.mn_range.len().divceil(self.r);
            Array3::from_shape_fn([panels, self.k_range.len(), self.r], |(panel, k, x)| {
                if self.mn_range.start + panel * self.r + x >= self.mn_range.end {
                    0
                } else {
                    let mn = panel * self.r + x + self.mn_range.start;
                    let k = k + self.k_range.start;
                    let coords = if self.is_a { (mn, k) } else { (k, mn) };
                    *input.get(coords).unwrap_or(&0)
                }
            })
        }

        fn check(&self) {
            assert_eq!(self.packer(), self.reference())
        }
    }

    impl Arbitrary for PackProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<PackProblem>;
        fn arbitrary_with(_args: ()) -> Self::Strategy {
            (any::<bool>(), 1usize..9, 1usize..20, 1usize..20)
                .prop_flat_map(|(is_a, r, k, mn)| {
                    (Just((is_a, r, k, mn)), sub_range_strat(0..k), sub_range_strat(0..mn))
                })
                .prop_map(|((is_a, r, k, mn), k_range, mn_range)| PackProblem {
                    k,
                    mn,
                    is_a,
                    r,
                    k_range,
                    mn_range,
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
        PackProblem { k: 2, mn: 1, is_a: false, r: 1, k_range: 0..2, mn_range: 0..1 }.check();
    }

    #[test]
    fn simple_b_2() {
        PackProblem { k: 2, mn: 2, is_a: false, r: 1, k_range: 0..2, mn_range: 0..2 }.check()
    }

    #[test]
    fn simple_b_3() {
        PackProblem { k: 2, mn: 1, is_a: false, r: 4, k_range: 0..2, mn_range: 0..1 }.check();
    }

    #[test]
    fn simple_a_1() {
        PackProblem { k: 2, mn: 2, is_a: true, r: 1, k_range: 0..2, mn_range: 0..2 }.check();
    }

    #[test]
    fn simple_a_2() {
        PackProblem { k: 2, mn: 3, is_a: true, r: 2, k_range: 0..2, mn_range: 0..3 }.check();
    }

    #[test]
    fn range_k_0() {
        PackProblem { k: 2, mn: 1, is_a: false, r: 1, k_range: 1..2, mn_range: 0..1 }.check();
    }

    #[test]
    fn range_k_1() {
        PackProblem { k: 2, mn: 2, is_a: false, r: 1, k_range: 0..2, mn_range: 0..1 }.check();
    }

    #[test]
    fn range_k_2() {
        PackProblem { k: 2, mn: 1, is_a: false, r: 6, k_range: 1..2, mn_range: 0..1 }.check();
    }

    #[test]
    fn range_mn_0() {
        PackProblem { k: 1, mn: 2, is_a: false, r: 2, k_range: 0..1, mn_range: 0..1 }.check();
    }

    #[test]
    fn range_b_4() {
        PackProblem { k: 1, mn: 2, is_a: false, r: 6, k_range: 0..1, mn_range: 1..2 }.check();
    }

    #[test]
    fn range_b_5() {
        PackProblem { k: 1, mn: 7, is_a: false, r: 6, k_range: 0..1, mn_range: 1..7 }.check();
    }
}
