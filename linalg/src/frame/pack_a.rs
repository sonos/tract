use std::fmt::Debug;

#[derive(Clone, Debug, Eq, PartialEq, Educe)]
#[educe(Hash)]
pub struct PackA {
    k: usize,
    m: usize,
    mr: usize,
    alignment: usize,
}

impl PackA {
    pub fn new(k: usize, m: usize, mr: usize, alignment: usize) -> PackA {
        PackA { k, m, mr, alignment }
    }
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn len(&self) -> usize {
        (self.m + self.mr - 1) / self.mr * self.mr * self.k
    }

    fn pack_panel_a<T: Copy>(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize, rows: usize) {
        let mr = self.mr;
        unsafe {
            for i in 0..self.k {
                for j in 0..rows {
                    *pa.offset((i * mr + j) as isize) =
                        *a.offset(i as isize * csa + j as isize * rsa)
                }
                #[cfg(debug_assertions)]
                {
                    for j in rows..self.mr {
                        *pa.offset((i * mr + j) as isize) = std::mem::zeroed();
                    }
                }
            }
        }
    }

    pub fn pack<T: Copy>(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize) {
        let mr = self.mr;
        assert!(pa as usize % self.alignment == 0);
        unsafe {
            for p in 0..(self.m / mr) {
                self.pack_panel_a(
                    pa.offset((p * mr * self.k) as isize),
                    a.offset((p * mr) as isize * rsa),
                    rsa,
                    csa,
                    mr,
                )
            }
            if self.m % mr != 0 {
                self.pack_panel_a(
                    pa.offset((self.m / mr * mr * self.k) as isize),
                    a.offset((self.m / mr * mr) as isize * rsa),
                    rsa,
                    csa,
                    self.m % mr,
                )
            }
        }
    }
}
