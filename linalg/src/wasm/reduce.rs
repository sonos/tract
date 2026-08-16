// SIMD reductions. LLVM does not vectorize these on wasm32: it will not
// reassociate a floating-point accumulator, so the generic forms keep a serial
// dependency chain one element long. Independent accumulators plus a
// tree-shaped horizontal reduction remove it. The max and min lanes are
// combined with `total_cmp` so the result matches the generic kernels exactly,
// NaN ordering included.

reduce_impl_wrap!(
    f32,
    wasm_max_f32_32n,
    32,
    4,
    (),
    f32::MIN,
    #[inline(never)]
    fn run(x: &[f32], _: ()) -> f32 {
        use std::arch::wasm32::*;
        unsafe {
            let len = x.len();
            let ptr = x.as_ptr() as *const v128;
            let mut acc = [f32x4_splat(f32::NEG_INFINITY); 8];
            let mut i = 0;
            while i + 32 <= len {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = f32x4_pmax(*a, v128_load(ptr.add(i / 4 + j)));
                }
                i += 32;
            }
            let a01 = f32x4_pmax(acc[0], acc[1]);
            let a23 = f32x4_pmax(acc[2], acc[3]);
            let a45 = f32x4_pmax(acc[4], acc[5]);
            let a67 = f32x4_pmax(acc[6], acc[7]);
            let s = f32x4_pmax(f32x4_pmax(a01, a23), f32x4_pmax(a45, a67));
            let mut m = f32x4_extract_lane::<0>(s);
            for v in
                [f32x4_extract_lane::<1>(s), f32x4_extract_lane::<2>(s), f32x4_extract_lane::<3>(s)]
            {
                if v.total_cmp(&m) == std::cmp::Ordering::Greater {
                    m = v;
                }
            }
            while i < len {
                if x[i].total_cmp(&m) == std::cmp::Ordering::Greater {
                    m = x[i];
                }
                i += 1;
            }
            m
        }
    },
    fn reduce_two(a: f32, b: f32) -> f32 {
        if a.total_cmp(&b) == std::cmp::Ordering::Greater { a } else { b }
    }
);

reduce_impl_wrap!(
    f32,
    wasm_min_f32_32n,
    32,
    4,
    (),
    f32::MAX,
    #[inline(never)]
    fn run(x: &[f32], _: ()) -> f32 {
        use std::arch::wasm32::*;
        unsafe {
            let len = x.len();
            let ptr = x.as_ptr() as *const v128;
            let mut acc = [f32x4_splat(f32::INFINITY); 8];
            let mut i = 0;
            while i + 32 <= len {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = f32x4_pmin(*a, v128_load(ptr.add(i / 4 + j)));
                }
                i += 32;
            }
            let a01 = f32x4_pmin(acc[0], acc[1]);
            let a23 = f32x4_pmin(acc[2], acc[3]);
            let a45 = f32x4_pmin(acc[4], acc[5]);
            let a67 = f32x4_pmin(acc[6], acc[7]);
            let s = f32x4_pmin(f32x4_pmin(a01, a23), f32x4_pmin(a45, a67));
            let mut m = f32x4_extract_lane::<0>(s);
            for v in
                [f32x4_extract_lane::<1>(s), f32x4_extract_lane::<2>(s), f32x4_extract_lane::<3>(s)]
            {
                if v.total_cmp(&m) == std::cmp::Ordering::Less {
                    m = v;
                }
            }
            while i < len {
                if x[i].total_cmp(&m) == std::cmp::Ordering::Less {
                    m = x[i];
                }
                i += 1;
            }
            m
        }
    },
    fn reduce_two(a: f32, b: f32) -> f32 {
        if a.total_cmp(&b) == std::cmp::Ordering::Less { a } else { b }
    }
);

reduce_impl_wrap!(
    f32,
    wasm_sum_f32_32n,
    32,
    4,
    (),
    0f32,
    #[inline(never)]
    fn run(x: &[f32], _: ()) -> f32 {
        use std::arch::wasm32::*;
        unsafe {
            let len = x.len();
            let ptr = x.as_ptr() as *const v128;
            let mut acc = [f32x4_splat(0f32); 8];
            let mut i = 0;
            while i + 32 <= len {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = f32x4_add(*a, v128_load(ptr.add(i / 4 + j)));
                }
                i += 32;
            }
            let a01 = f32x4_add(acc[0], acc[1]);
            let a23 = f32x4_add(acc[2], acc[3]);
            let a45 = f32x4_add(acc[4], acc[5]);
            let a67 = f32x4_add(acc[6], acc[7]);
            let s = f32x4_add(f32x4_add(a01, a23), f32x4_add(a45, a67));
            let mut sum = f32x4_extract_lane::<0>(s)
                + f32x4_extract_lane::<1>(s)
                + f32x4_extract_lane::<2>(s)
                + f32x4_extract_lane::<3>(s);
            while i < len {
                sum += x[i];
                i += 1;
            }
            sum
        }
    },
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

map_reduce_impl_wrap!(
    f32,
    wasm_softmax2_fastcompact_f32_32n,
    32,
    4,
    f32,
    f32::MIN,
    0f32,
    #[inline(never)]
    fn run(buf: &mut [f32], max: f32) -> f32 {
        use std::arch::wasm32::*;
        const SLOPE: f32 = 8388608.0f32 / 0.6931471805f32;
        const OFFSET: f32 = 1065353216.0f32 - 60801.0f32;
        unsafe {
            let sl = f32x4_splat(SLOPE);
            let of = f32x4_splat(OFFSET);
            let mx = f32x4_splat(max);
            let mut acc = [f32x4_splat(0f32); 16];
            let len = buf.len();
            let p = buf.as_mut_ptr();
            let blocks = len / 64;
            for i in 0..blocks {
                let b = p.add(i * 64);
                for (j, a) in acc.iter_mut().enumerate() {
                    let q = b.add(j * 4);
                    let d = f32x4_sub(v128_load(q as *const v128), mx);
                    let e = u32x4_trunc_sat_f32x4(madd_f32x4!(of, d, sl));
                    v128_store(q as *mut v128, e);
                    *a = f32x4_add(*a, e);
                }
            }
            let h0 = f32x4_add(f32x4_add(acc[0], acc[1]), f32x4_add(acc[2], acc[3]));
            let h1 = f32x4_add(f32x4_add(acc[4], acc[5]), f32x4_add(acc[6], acc[7]));
            let h2 = f32x4_add(f32x4_add(acc[8], acc[9]), f32x4_add(acc[10], acc[11]));
            let h3 = f32x4_add(f32x4_add(acc[12], acc[13]), f32x4_add(acc[14], acc[15]));
            let s = f32x4_add(f32x4_add(h0, h1), f32x4_add(h2, h3));
            let mut sum = f32x4_extract_lane::<0>(s)
                + f32x4_extract_lane::<1>(s)
                + f32x4_extract_lane::<2>(s)
                + f32x4_extract_lane::<3>(s);
            for v in buf[blocks * 64..].iter_mut() {
                let y = f32::from_bits(((SLOPE * (*v - max)) + OFFSET) as u32);
                *v = y;
                sum += y;
            }
            sum
        }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

#[cfg(test)]
mod test_max {
    use super::*;
    crate::max_frame_tests!(true, f32, wasm_max_f32_32n);
}

#[cfg(test)]
mod test_min {
    use super::*;
    crate::min_frame_tests!(true, f32, wasm_min_f32_32n);
}

#[cfg(test)]
mod test_sum {
    use super::*;
    crate::sum_frame_tests!(true, f32, wasm_sum_f32_32n);
}

#[cfg(test)]
mod test_softmax {
    use super::*;
    crate::softmax_l2_frame_tests!(true, f32, wasm_softmax2_fastcompact_f32_32n);
}

/// RMS-normalises `buf` in place: each element is divided by the root of the
/// mean square plus `eps`. The sum of squares keeps sixteen independent
/// accumulators so the multiply-accumulate latency does not serialise, which is
/// what the generic scalar form cannot avoid.
pub fn rms_norm_f32(buf: &mut [f32], eps: f32) {
    use std::arch::wasm32::*;
    if buf.is_empty() {
        return;
    }
    unsafe {
        let len = buf.len();
        let ptr = buf.as_mut_ptr();
        let chunks = len / 64;
        let mut acc = [f32x4_splat(0f32); 16];
        for i in 0..chunks {
            let base = ptr.add(i * 64);
            for (j, a) in acc.iter_mut().enumerate() {
                let v = v128_load(base.add(j * 4) as *const v128);
                *a = madd_f32x4!(*a, v, v);
            }
        }
        let mut pairs = [f32x4_splat(0f32); 8];
        for (k, p) in pairs.iter_mut().enumerate() {
            *p = f32x4_add(acc[2 * k], acc[2 * k + 1]);
        }
        let q0 = f32x4_add(f32x4_add(pairs[0], pairs[1]), f32x4_add(pairs[2], pairs[3]));
        let q1 = f32x4_add(f32x4_add(pairs[4], pairs[5]), f32x4_add(pairs[6], pairs[7]));
        let s = f32x4_add(q0, q1);
        let mut sum = f32x4_extract_lane::<0>(s)
            + f32x4_extract_lane::<1>(s)
            + f32x4_extract_lane::<2>(s)
            + f32x4_extract_lane::<3>(s);
        for i in chunks * 64..len {
            let v = *ptr.add(i);
            sum += v * v;
        }

        let scale = 1f32 / (sum / len as f32 + eps).sqrt();
        let scale_v = f32x4_splat(scale);
        for i in 0..chunks {
            let base = ptr.add(i * 64);
            for j in 0..16 {
                let q = base.add(j * 4);
                v128_store(q as *mut v128, f32x4_mul(v128_load(q as *const v128), scale_v));
            }
        }
        for i in chunks * 64..len {
            *ptr.add(i) *= scale;
        }
    }
}
