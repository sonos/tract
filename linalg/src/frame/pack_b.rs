use std::marker::PhantomData;
use tract_data::internal::*;

#[derive(Clone, Debug, Eq, PartialEq, Educe)]
#[educe(Hash)]
pub struct PackB {
    k: usize,
    n: usize,
    nr: usize,
    alignment: usize,
    end_padding_record: usize,
}

impl PackB {
    pub fn new(k: usize, n: usize, nr: usize, alignment: usize, end_padding_record: usize) -> PackB {
        PackB { k, n, nr, alignment, end_padding_record }
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn len(&self) -> usize {
        (self.n + self.nr - 1) / self.nr * self.nr * self.k + self.end_padding_record * self.nr
    }

    fn pack_panel_b_t<T: Copy>(
        &self,
        pb: *mut T,
        b: *const T,
        rsb: isize,
        csb: isize,
        cols: usize,
    ) {
        let nr = self.nr;
        unsafe {
            for i in 0..self.k {
                for j in 0..cols {
                    *pb.offset((i * nr + j) as isize) =
                        *b.offset(j as isize * csb + i as isize * rsb)
                }
                #[cfg(debug_assertions)]
                {
                    for j in cols..self.nr {
                        *pb.offset((i * nr + j) as isize) = std::mem::zeroed();
                    }
                }
            }
        }
    }

    fn pack_panel_b(
        &self,
        dt: DatumType,
        pb: *mut u8,
        b: *const u8,
        rsb: isize,
        csb: isize,
        cols: usize,
    ) {
        dispatch_copy_by_size!(Self::pack_panel_b_t(dt)(self, pb as _, b as _, rsb, csb, cols))
    }

    pub unsafe fn pack<'a, 'b>(
        &self,
        mut pb: impl std::borrow::BorrowMut<TensorView<'a>>,
        b: impl std::borrow::Borrow<TensorView<'b>>,
        trans: bool,
    ) {
        let pb = pb.borrow_mut();
        let b = b.borrow();
        let dt = pb.datum_type();
        let (rsb, csb) = if trans {
            (1, b.shape()[b.rank() - 1] as isize)
        } else {
            (b.shape()[b.rank() - 1] as isize, 1)
        };
        let pb = pb.as_ptr_mut_unchecked::<u8>();
        let b = b.as_ptr_unchecked::<u8>();
        let nr = self.nr;
        assert!(pb as usize % self.alignment == 0);
        for p in 0..(self.n / nr) {
            self.pack_panel_b(
                dt,
                pb.offset((p * nr * self.k * dt.size_of()) as isize),
                b.offset((p * nr * dt.size_of()) as isize * csb),
                rsb,
                csb,
                nr,
            )
        }
        if self.n % nr != 0 {
            self.pack_panel_b(
                dt,
                pb.offset((self.n / nr * nr * self.k * dt.size_of()) as isize),
                b.offset((self.n / nr * nr * dt.size_of()) as isize * csb),
                rsb,
                csb,
                self.n % nr,
            )
        }
    }

    pub fn write_packed_by_rows<'p, T: Copy>(&self, pb: &'p mut [T]) -> PackedWriter<'p, T> {
        PackedWriter::new(pb, self.nr, self.n, self.k)
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
