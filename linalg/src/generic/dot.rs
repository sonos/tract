use std::ops::*;
use crate::f16;

extern "C" {
    fn dot_f16(sum: *mut f16, count: usize, iptr: *const f16, kptr: *const f16, ioffsets: *const isize, koffsets: *const isize);
}

pub unsafe fn dotprod_f16(
    sum: *mut f16,
    iptr: *const f16,
    kptr: *const f16,
    offsets: *const Box<[(usize, isize)]>,
    input_center_offset: isize,
) {
    let count = (*offsets).len();
    // If we could guarantee value_offsets was a repr(C) tuple struct, we can avoid these allocations
    let (koffsets, ioffsets): (Vec<isize>, Vec<isize>) = (*offsets).iter().map(move |pair| (pair.0 as isize, pair.1 + input_center_offset)).unzip();
    dot_f16(sum, count, iptr, kptr, ioffsets.as_ptr(), koffsets.as_ptr());
}

// Generic fallback implementation of dotprod
pub unsafe fn dotprod<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy>(
    sum: *mut T,
    iptr: *const T,
    kptr: *const T,
    offsets: *const Box<[(usize, isize)]>,
    input_center_offset: isize,
) {
    let count = (*offsets).len();
    if count == 4 {
        let (ix, v) = &(*offsets)[0];
        let k0 = *kptr.add(*ix);
        let i0 = *iptr.offset(*v + input_center_offset);
        let (ix, v) = &(*offsets)[1];
        let k1 = *kptr.add(*ix);
        let i1 = *iptr.offset(*v + input_center_offset);
        let (ix, v) = &(*offsets)[2];
        let k2 = *kptr.add(*ix);
        let i2 = *iptr.offset(*v + input_center_offset);
        let (ix, v) = &(*offsets)[3];
        let k3 = *kptr.add(*ix);
        let i3 = *iptr.offset(*v + input_center_offset);
        *sum = *sum + k0 * i0 + k1 * i1 + k2 * i2 + k3 * i3;
    } else if count == 3 {
        let (ix, v) = &(*offsets)[0];
        let k0 = *kptr.add(*ix);
        let i0 = *iptr.offset(*v + input_center_offset);
        let (ix, v) = &(*offsets)[1];
        let k1 = *kptr.add(*ix);
        let i1 = *iptr.offset(*v + input_center_offset);
        let (ix, v) = &(*offsets)[2];
        let k2 = *kptr.add(*ix);
        let i2 = *iptr.offset(*v + input_center_offset);
        *sum = *sum + k0 * i0 + k1 * i1 + k2 * i2;
    } else {
        for idx in 0..count {
            let (ix, v) = &(*offsets)[idx];
            let k = *kptr.add(*ix);
            let i = *iptr.offset(*v + input_center_offset);
            *sum = *sum + k * i;
        }
    }
}