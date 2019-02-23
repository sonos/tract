use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, Mul};

use std::marker::PhantomData;

pub trait Conv<T: Copy + Add + Mul + Zero + Debug>: Send + Sync + Debug + objekt::Clone {
    fn packed_a_len(&self) -> usize;
    fn packed_a_alignment(&self) -> usize;
    fn pack_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize);

    fn conv(&self, pa: *const T, b: *const T, c: *mut T, rsc: isize, csc: isize);
}

clone_trait_object!(<T> Conv<T> where T: Copy + Add + Mul + Zero);

pub trait ConvKer<T: Copy + Add + Mul + Zero>: Copy + Clone + Debug + Send + Sync {
    #[inline(always)]
    fn name() -> &'static str;
    #[inline(always)]
    fn kernel(
        k: usize,
        a: *const T,
        b_tops: *const *const T,
        b_down_offsets: *const isize,
        c: *mut T,
        rsc: usize,
        csc: usize,
    );
    #[inline(always)]
    fn mr() -> usize;
    #[inline(always)]
    fn nr() -> usize;
    #[inline(always)]
    fn alignment_bytes_a() -> usize;
    #[inline(always)]
    fn alignment_bytes_b() -> usize;
}

/// filters: O IHW packed as for matmul
/// "m" = O
/// "k" = I * Kh * Kw
///
/// data: unpacked

#[derive(Clone)]
pub struct PackedConv<K, T>
where
    K: ConvKer<T> + Debug,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    co: usize,
    kernel_offsets: Vec<isize>,
    n: usize,
    data_offsets: Vec<isize>,
    _kernel: PhantomData<(K, T)>,
}

impl<K, T> std::fmt::Debug for PackedConv<K, T>
where
    K: ConvKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            fmt,
            "Conv co:{} k:{} data:{} ({} {}x{})",
            self.co,
            self.kernel_offsets.len(),
            self.data_offsets.len(),
            K::name(),
            K::mr(),
            K::nr()
        )
    }
}

impl<K, T> PackedConv<K, T>
where
    K: ConvKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    pub fn new(
        co: usize,
        kernel_offsets: Vec<isize>,
        mut data_offsets: Vec<isize>,
    ) -> PackedConv<K, T> {
        assert!(data_offsets.len() > 0);
        assert!(kernel_offsets.len() > 0);
        let n = data_offsets.len();
        while data_offsets.len() % K::nr() != 0 {
            data_offsets.push(data_offsets[data_offsets.len() - 1]);
        }
        PackedConv {
            co,
            kernel_offsets,
            n,
            data_offsets,
            _kernel: PhantomData,
        }
    }

    fn pack_panel_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize, rows: usize) {
        let mr = K::mr();
        for i in 0..self.kernel_offsets.len() {
            for j in 0..rows {
                unsafe {
                    *pa.offset((i * mr + j) as isize) =
                        *a.offset(i as isize * csa + j as isize * rsa)
                }
            }
        }
    }
}

impl<K, T> Conv<T> for PackedConv<K, T>
where
    K: ConvKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync + PartialEq,
{
    fn packed_a_alignment(&self) -> usize {
        K::alignment_bytes_a()
    }

    fn packed_a_len(&self) -> usize {
        let mr = K::mr();
        (self.co + mr - 1) / mr * mr * self.kernel_offsets.len()
    }

    fn pack_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize) {
        let mr = K::mr();
        assert!(pa as usize % K::alignment_bytes_a() == 0);
        unsafe {
            for p in 0..(self.co / mr) {
                self.pack_panel_a(
                    pa.offset((p * mr * self.kernel_offsets.len()) as isize),
                    a.offset((p * mr) as isize * rsa),
                    rsa,
                    csa,
                    mr,
                )
            }
            if self.co % mr != 0 {
                self.pack_panel_a(
                    pa.offset((self.co / mr * mr * self.kernel_offsets.len()) as isize),
                    a.offset((self.co / mr * mr) as isize * rsa),
                    rsa,
                    csa,
                    self.co % mr,
                )
            }
            assert_eq!(*pa, *a);
        }
    }

    fn conv(&self, pa: *const T, b: *const T, c: *mut T, rsc: isize, csc: isize) {
        assert!(pa as usize % K::alignment_bytes_a() == 0);
        let mr = K::mr();
        let nr = K::nr();
        let co = self.co;
        let k = self.kernel_offsets.len();
        let n = self.n;
        let mut tmpc = vec![T::zero(); mr * nr];
        unsafe {
            let btops: Vec<*const T> = self.data_offsets.iter().map(|&o| b.offset(o)).collect();
            for ia in 0..co / mr {
                for ib in 0..n / nr {
                    K::kernel(
                        k,
                        pa.offset((ia * k * mr) as isize),
                        btops.as_ptr().offset((ib * nr) as isize),
                        self.kernel_offsets.as_ptr().offset((ia * mr) as isize),
                        c.offset((mr * ia) as isize * rsc + (nr * ib) as isize * csc),
                        rsc as usize,
                        csc as usize,
                    );
                }
                if n % nr != 0 {
                    K::kernel(
                        k,
                        pa.offset((ia * k * mr) as isize),
                        btops.as_ptr().offset((n / nr * nr) as isize),
                        self.kernel_offsets.as_ptr().offset((ia * mr) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                        1,
                    );
                    for y in 0..mr {
                        for x in 0..(n % nr) {
                            *c.offset(
                                (mr * ia + y) as isize * rsc + (x + n / nr * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
            }
            if co % mr != 0 {
                for ib in 0..n / nr {
                    K::kernel(
                        k,
                        pa.offset((co / mr * mr * k) as isize),
                        btops.as_ptr().offset((ib * nr) as isize),
                        self.kernel_offsets.as_ptr().offset((co / mr * mr) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                        1,
                    );
                    for y in 0..(co % mr) {
                        for x in 0..nr {
                            *c.offset(
                                (y + co / mr * mr) as isize * rsc + (x + ib * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
                if n % nr != 0 {
                    K::kernel(
                        k,
                        pa.offset((co / mr * mr * k) as isize),
                        btops.as_ptr().offset((n / nr * nr) as isize),
                        self.kernel_offsets.as_ptr().offset((co / mr * mr) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                        1,
                    );
                    for y in 0..(co % mr) {
                        for x in 0..(n % nr) {
                            *c.offset(
                                (y + co / mr * mr) as isize * rsc
                                    + (x + n / nr * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::align;
    use proptest::prelude::*;
    use proptest::*;

    /*
    pub fn strat_ker_mat_mul<MM: PackedMatMulKer<f32>>(
    ) -> BoxedStrategy<(usize, Vec<f32>, Vec<f32>)> {
        (0usize..35)
            .prop_flat_map(move |k| {
                (
                    Just(k),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), MM::mr() * k),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), MM::nr() * k),
                )
            })
            .boxed()
    }

    pub fn test_ker_mat_mul<MM: PackedMatMulKer<f32>>(
        k: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        let pa = align::realign_slice(a, MM::alignment_bytes_a());
        let pb = align::realign_slice(b, MM::alignment_bytes_b());

        let mut expect = vec![0.0f32; MM::mr() * MM::nr()];
        for x in 0..MM::nr() {
            for y in 0..MM::mr() {
                expect[x + y * MM::nr()] = (0..k)
                    .map(|k| pa[k * MM::mr() + y] * pb[k * MM::nr() + x])
                    .sum::<f32>();
            }
        }
        let mut found = vec![9999.0f32; expect.len()];
        MM::kernel(k, pa.as_ptr(), pb.as_ptr(), found.as_mut_ptr(), MM::nr(), 1);
        prop_assert_eq!(found, expect);
        Ok(())
    }

    */

    /// ci, co, kt, stride, dilation, filters, data
    pub fn strat_conv_1d() -> BoxedStrategy<(usize, usize, usize, usize, usize, Vec<f32>, Vec<f32>)> {
        (
            1usize..5,
            1usize..5,
            1usize..5,
            1usize..5,
            1usize..5,
            1usize..25,
        )
            .prop_filter(
                "data must at least as wide as kernel",
                |(_ci, _co, kt, _stride, dilation, t)| (kt - 1) * dilation + 1 <= *t,
            )
            .prop_flat_map(move |(ci, co, kt, stride, dilation, t)| {
                (
                    Just(ci),
                    Just(co),
                    Just(kt),
                    Just(stride),
                    Just(dilation),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), ci * co * kt),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), t * ci),
                )
            })
            .boxed()
    }

    pub fn test_conv_1d_f32<C: Conv<f32>>(
        conv: C,
        ci: usize,
        co: usize,
        kt: usize,
        stride: usize,
        dilation: usize,
        filters: &[f32],
        data: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        unsafe {
            let k = ci * kt;
            let kernel_field = dilation * (kt - 1) + 1;
            let t = data.len() / ci;
            let n = (t - kernel_field) / stride + 1;

            let mut expect = vec![0.0f32; co * n];
            for x in 0..n {
                for ico in 0..co {
                    for ikt in 0..kt {
                        for ici in 0..ci {
                            let f = filters[ici * kt + ikt + k * ico];
                            let d = data[x * stride + ikt * dilation + ici * t];
                            expect[x + ico * n] += f * d;
                        }
                    }
                }
            }

            let mut packed_a: Vec<f32> =
                align::uninitialized(conv.packed_a_len(), conv.packed_a_alignment());
            conv.pack_a(packed_a.as_mut_ptr(), filters.as_ptr(), k as isize, 1);

            let mut found = vec![9999.0f32; co * n];
            conv.conv(packed_a.as_ptr(), data.as_ptr(), found.as_mut_ptr(), n as isize, 1);

            prop_assert_eq!(found, expect);
        }
        Ok(())
    }

}
