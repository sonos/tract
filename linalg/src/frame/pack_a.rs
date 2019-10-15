use num_traits::Zero;
use std::fmt::Debug;
use std::marker::PhantomData;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackA<T: Copy + Zero> {
    k: usize,
    m: usize,
    mr: usize,
    alignment: usize,
    _boo: PhantomData<T>,
}

impl<T: Copy + Zero + Debug> PackA<T> {
    pub fn new(k: usize, m: usize, mr: usize, alignment: usize) -> PackA<T> {
        PackA { k, m, mr, alignment, _boo: PhantomData }
    }
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn len(&self) -> usize {
        (self.m + self.mr - 1) / self.mr * self.mr * self.k
    }

    fn pack_panel_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize, rows: usize) {
        let mr = self.mr;
        for i in 0..self.k {
            for j in 0..rows {
                unsafe {
                    *pa.offset((i * mr + j) as isize) =
                        *a.offset(i as isize * csa + j as isize * rsa)
                }
            }
            #[cfg(debug_assertions)]
            for j in rows..mr {
                unsafe {
                    *pa.offset((i * mr + j) as isize) = T::zero();
                }
            }
        }
    }

    pub fn pack(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize) {
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
