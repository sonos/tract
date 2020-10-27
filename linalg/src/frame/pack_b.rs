use std::marker::PhantomData;

#[derive(Clone, Debug, Eq, PartialEq, Educe)]
#[educe(Hash)]
pub struct PackB {
    k: usize,
    n: usize,
    nr: usize,
    alignment: usize,
}

impl PackB {
    pub fn new(k: usize, n: usize, nr: usize, alignment: usize) -> PackB {
        PackB { k, n, nr, alignment }
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn len(&self) -> usize {
        (self.n + self.nr - 1) / self.nr * self.nr * self.k
    }

    pub fn pack<T: Copy>(&self, pb: *mut T, b: *const T, rsb: isize, csb: isize) {
        let nr = self.nr;
        assert!(pb as usize % self.alignment == 0);
        unsafe {
            for p in 0..(self.n / nr) {
                self.pack_panel_b(
                    pb.offset((p * nr * self.k) as isize),
                    b.offset((p * nr) as isize * csb),
                    rsb,
                    csb,
                    nr,
                )
            }
            if self.n % nr != 0 {
                self.pack_panel_b(
                    pb.offset((self.n / nr * nr * self.k) as isize),
                    b.offset((self.n / nr * nr) as isize * csb),
                    rsb,
                    csb,
                    self.n % nr,
                )
            }
        }
    }

    fn pack_panel_b<T: Copy>(&self, pb: *mut T, b: *const T, rsb: isize, csb: isize, cols: usize) {
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
