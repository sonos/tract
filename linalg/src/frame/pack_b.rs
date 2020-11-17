use std::marker::PhantomData;
use tract_data::internal::*;

#[derive(Clone, Debug, Eq, PartialEq, Educe)]
#[educe(Hash)]
pub struct PackB {
    k: usize,
    r: usize,
    alignment: usize,
    end_padding_record: usize,
}

impl PackB {
    pub fn new(k: usize, nr: usize, alignment: usize, end_padding_record: usize) -> PackB {
        PackB { k, r: nr, alignment, end_padding_record }
    }

    pub fn alignment(&self) -> usize {
        self.alignment
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
        let mut packer = self.write_packed_by_rows(pb, mn);
        if mn_stride == 1 {
            for k in 0..self.k as isize {
                for x in 0..mn as isize {
                    packer.write(*b.get_unchecked((x + k_stride * k) as usize))
                }
            }
        } else if k_stride == 1 {
            for k in 0..self.k as isize {
                for x in 0..mn as isize {
                    packer.write(*b.get_unchecked((x * mn_stride + k) as usize))
                }
            }
        } else {
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

    pub fn write_packed_by_rows<'p, T: Copy>(
        &self,
        pb: &'p mut [T],
        mn: usize,
    ) -> PackedWriter<'p, T> {
        PackedWriter::new(pb, self.r, mn, self.k)
    }
}

#[derive(Debug)]
pub struct PackedWriter<'p, T>
where
    T: Copy,
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

impl<'p, T> PackedWriter<'p, T>
where
    T: Copy,
{
    pub fn new(data: &'p mut [T], panel_width: usize, mn: usize, k: usize) -> PackedWriter<'p, T> {
        let panels = (mn + panel_width - 1) / panel_width;
        let last_panel_width = mn - (panels - 1) * panel_width;
        PackedWriter {
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
