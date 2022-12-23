use std::ops::*;

pub unsafe fn dotprod<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy>(
    sum: *mut T,
    iptr: *const T,
    kptr: *const T,
    offsets: *const Box<[(usize, isize)]>,
    input_center_offset: isize,
) {
    if (*offsets).len() == 4 {
        let (ix, v) = &(*offsets)[0];
        let k0 = *kptr.add(*ix);
        let i0 = *iptr.offset(*v as isize + input_center_offset);
        let (ix, v) = &(*offsets)[1];
        let k1 = *kptr.add(*ix);
        let i1 = *iptr.offset(*v as isize + input_center_offset);
        let (ix, v) = &(*offsets)[2];
        let k2 = *kptr.add(*ix);
        let i2 = *iptr.offset(*v as isize + input_center_offset);
        let (ix, v) = &(*offsets)[3];
        let k3 = *kptr.add(*ix);
        let i3 = *iptr.offset(*v as isize + input_center_offset);
        *sum = *sum + k0 * i0 + k1 * i1 + k2 * i2 + k3 * i3;
    } else if (*offsets).len() == 3 {
        let (ix, v) = &(*offsets)[0];
        let k0 = *kptr.add(*ix);
        let i0 = *iptr.offset(*v as isize + input_center_offset);
        let (ix, v) = &(*offsets)[1];
        let k1 = *kptr.add(*ix);
        let i1 = *iptr.offset(*v as isize + input_center_offset);
        let (ix, v) = &(*offsets)[2];
        let k2 = *kptr.add(*ix);
        let i2 = *iptr.offset(*v as isize + input_center_offset);
        *sum = *sum + k0 * i0 + k1 * i1 + k2 * i2;
    } else {
        for idx in 0..(*offsets).len() {
            let (ix, v) = &(*offsets)[idx];
            let k = *kptr.add(*ix);
            let i = *iptr.offset(*v as isize + input_center_offset);
            *sum = *sum + k * i;
        }
    }
}
