use std::fmt::Debug;
use tract_data::internal::*;

#[derive(Clone, Debug, Eq, PartialEq, Educe)]
#[educe(Hash)]
pub struct PackA {
    k: usize,
    m: usize,
    mr: usize,
    alignment: usize,
    end_padding_record: usize,
}

impl PackA {
    pub fn new(k: usize, m: usize, mr: usize, alignment: usize, end_padding_record: usize) -> PackA {
        PackA { k, m, mr, alignment, end_padding_record }
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn len(&self) -> usize {
        (self.m + self.mr - 1) / self.mr * self.mr * self.k + self.end_padding_record * self.mr
    }

    fn pack_panel_a_t<T: Datum + Copy>(
        &self,
        pa: *mut T,
        a: *const T,
        rsa: isize,
        csa: isize,
        rows: usize,
    ) {
        let mr = self.mr;
        unsafe {
            for k in 0..self.k {
                for m in 0..rows {
                    *pa.offset((k * mr + m) as isize) =
                        *a.offset(k as isize * csa + m as isize * rsa)
                }
                #[cfg(debug_assertions)]
                {
                    for m in rows..self.mr {
                        *pa.offset((k * mr + m) as isize) = std::mem::zeroed();
                    }
                }
            }
        }
    }

    fn pack_panel_a(
        &self,
        dt: DatumType,
        pa: *mut u8,
        a: *const u8,
        rsa: isize,
        csa: isize,
        rows: usize,
    ) {
        dispatch_copy_by_size!(Self::pack_panel_a_t(dt)(self, pa as _, a as _, rsa, csa, rows))
    }

    pub unsafe fn pack<'a>(
        &self,
        mut pa: impl std::borrow::BorrowMut<TensorView<'a>>,
        a: impl std::borrow::Borrow<TensorView<'a>>,
        trans: bool,
    ) {
        let pa = pa.borrow_mut();
        let a = a.borrow();
        let dt = pa.datum_type();
        let (rsa, csa) = if trans {
            (1, a.shape()[a.rank() - 1] as isize)
        } else {
            (a.shape()[a.rank() - 1] as isize, 1)
        };
        let pa = pa.as_ptr_mut_unchecked::<u8>();
        let a = a.as_ptr_unchecked::<u8>();
        let mr = self.mr;
        assert!(pa as usize % self.alignment == 0);
        for p in 0..(self.m / mr) {
            self.pack_panel_a(
                dt,
                pa.offset((p * mr * self.k * dt.size_of()) as isize),
                a.offset((p * mr * dt.size_of()) as isize * rsa),
                rsa,
                csa,
                mr,
            )
        }
        if self.m % mr != 0 {
            self.pack_panel_a(
                dt,
                pa.offset((self.m / mr * mr * self.k * dt.size_of()) as isize),
                a.offset((self.m / mr * mr * dt.size_of()) as isize * rsa),
                rsa,
                csa,
                self.m % mr,
            )
        }
    }
}
