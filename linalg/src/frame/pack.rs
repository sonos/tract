use std::fmt::Debug;
use std::marker::PhantomData;
use tract_data::internal::*;

#[derive(Clone, Debug, Eq, PartialEq, Educe)]
#[educe(Hash)]
pub struct Packer {
    k: usize,
    r: usize,
    alignment: usize,
    end_padding_record: usize,
}

impl Packer {
    pub fn new(k: usize, nr: usize, alignment: usize, end_padding_record: usize) -> Packer {
        Packer { k, r: nr, alignment, end_padding_record }
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn panel_width(&self) -> usize {
        self.r
    }

    pub fn len(&self, n: usize) -> usize {
        (n + self.r - 1) / self.r * self.r * self.k + self.end_padding_record * self.r
    }

    unsafe fn pack_t<'p, 'i, T: Datum + Copy>(
        &self,
        pb: &mut TensorView<'p>,
        b: &TensorView<'i>,
        mn: usize,
        k_stride: isize,
        mn_stride: isize,
    ) {
        let pb = pb.as_slice_mut_unchecked::<T>();
        let b = b.as_slice_unchecked::<T>();
        #[cfg(debug_assertions)]
        {
            pb.iter_mut().for_each(|v| *v = T::default());
        }
        if mn_stride == 1 {
            let mut packer = self.write_with_k_outer(pb, mn);
            for k in 0..self.k as isize {
                for x in 0..mn as isize {
                    packer.write(*b.get_unchecked((x + k_stride * k) as usize))
                }
            }
        } else if k_stride == 1 {
            let mut packer = self.write_with_k_inner(pb, mn);
            for x in 0..mn as isize {
                for k in 0..self.k as isize {
                    packer.write(*b.get_unchecked((x * mn_stride + k) as usize))
                }
            }
        } else {
            let mut packer = self.write_with_k_outer(pb, mn);
            for k in 0..self.k as isize {
                for x in 0..mn as isize {
                    packer.write(*b.get_unchecked((x * mn_stride + k_stride * k) as usize))
                }
            }
        }
    }

    pub unsafe fn pack<'a, 'b>(
        &self,
        mut pb: impl std::borrow::BorrowMut<TensorView<'a>>,
        b: impl std::borrow::Borrow<TensorView<'b>>,
        k_axis: usize,
        mn_axis: usize,
    ) {
        let pb = pb.borrow_mut();
        let b = b.borrow();
        debug_assert_eq!(b.shape()[k_axis], self.k);
        debug_assert_eq!(pb.len(), self.len(b.shape()[mn_axis]));
        let dt = pb.datum_type();
        dispatch_copy!(Self::pack_t(dt)(
            self,
            pb,
            b,
            b.shape()[mn_axis],
            b.strides()[k_axis],
            b.strides()[mn_axis]
        ));
    }

    pub fn write_with_k_outer<'p, T: Copy + Debug>(
        &self,
        pb: &'p mut [T],
        mn: usize,
    ) -> KOutWriter<'p, T> {
        KOutWriter::new(pb, self.r, mn, self.k)
    }

    pub fn write_with_k_inner<'p, T: Copy + Debug>(
        &self,
        pb: &'p mut [T],
        mn: usize,
    ) -> KInWriter<'p, T> {
        KInWriter::new(pb, self.r, mn, self.k)
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
    pub fn new(data: &'p mut [T], panel_width: usize, mn: usize, k: usize) -> KOutWriter<'p, T> {
        let panels = (mn + panel_width - 1) / panel_width;
        let last_panel_width = mn - (panels - 1) * panel_width;
        KOutWriter {
            ptr: data.as_mut_ptr(),
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

    #[inline]
    pub fn write(&mut self, t: T) {
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
    pub fn new(data: &'p mut [T], panel_width: usize, mn: usize, k: usize) -> KInWriter<'p, T> {
        let panels = (mn + panel_width - 1) / panel_width;
        let last_panel_width = mn - (panels - 1) * panel_width;
        KInWriter {
            ptr: data.as_mut_ptr(),
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

    #[inline]
    pub fn write(&mut self, t: T) {
        unsafe {
            *self.ptr = t;
            self.remain_on_k -= 1;
            self.ptr = self.ptr.offset(self.panel_width as isize);
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

#[cfg(test)]
mod test {
    use proptest::prelude::*;
    use tract_data::internal::*;
    use tract_ndarray::prelude::*;

    #[derive(Debug)]
    struct PackProblem {
        k: usize,
        mn: usize,
        is_a: bool,
        r: usize,
        input: Array2<u32>,
    }

    impl PackProblem {
        fn packer(&self) -> Vec<u32> {
            let packer = super::Packer::new(self.k, self.r, 1, 0);
            let input = self.input.clone().into_tensor();
            let mut output = Tensor::zero::<u32>(&[packer.len(self.mn)]).unwrap();
            unsafe {
                packer.pack(
                    output.view_mut(),
                    input.view(),
                    self.is_a as usize,
                    !self.is_a as usize,
                )
            };
            output.as_slice::<u32>().unwrap().to_vec()
        }

        fn reference(&self) -> Vec<u32> {
            let panels = self.mn.div_ceil(self.r);
            let len = panels * self.k * self.r;
            let mut vec = vec![0; len];
            for panel in 0..panels {
                for k in 0..self.k {
                    for x in 0..self.r {
                        let ix = panel * self.r + x;
                        let v = *self
                            .input
                            .get(if self.is_a { (ix, k) } else { (k, ix) })
                            .unwrap_or(&0);
                        vec[panel * self.k * self.r + k * self.r + x] = v;
                    }
                }
            }
            vec
        }
    }

    impl Arbitrary for PackProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<PackProblem>;
        fn arbitrary_with(_args: ()) -> Self::Strategy {
            (any::<bool>(), 1usize..4, 1usize..8, 1usize..8)
                .prop_flat_map(|(is_a, r, mn, k)| {
                    (Just((is_a, r, mn, k)), proptest::collection::vec(0u32..40, mn * k..=mn * k))
                })
                .prop_map(|((is_a, r, mn, k), input)| PackProblem {
                    k,
                    mn,
                    is_a,
                    r,
                    input: arr1(&*input).into_shape(if is_a { (mn, k) } else { (k, mn) }).unwrap(),
                })
                .boxed()
        }
    }

    proptest::proptest! {
        #[test]
        fn prop(pb in any::<PackProblem>()) {
            assert_eq!(pb.reference(), pb.packer());
        }
    }

    #[test]
    fn simple_b_1() {
        let pb = PackProblem { k: 2, mn: 1, is_a: false, r: 1, input: arr2(&[[0], [1]]) };
        assert_eq!(pb.reference(), pb.packer());
    }

    #[test]
    fn simple_b_2() {
        let pb = PackProblem { k: 2, mn: 2, is_a: false, r: 1, input: arr2(&[[0, 0], [0, 1]]) };
        assert_eq!(pb.reference(), pb.packer());
    }

    #[test]
    fn simple_a_1() {
        let pb = PackProblem { k: 2, mn: 2, is_a: true, r: 1, input: arr2(&[[0, 0], [0, 1]]) };
        assert_eq!(pb.reference(), pb.packer());
    }

    #[test]
    fn simple_a_2() {
        let pb =
            PackProblem { k: 2, mn: 3, is_a: true, r: 2, input: arr2(&[[0, 0], [0, 0], [0, 1]]) };
        assert_eq!(pb.reference(), pb.packer());
    }
}
