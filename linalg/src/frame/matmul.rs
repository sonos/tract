use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, Mul};

use std::marker::PhantomData;

use super::PackB;

pub trait MatMul<T: Copy + Add + Mul + Zero + Debug>: Send + Sync + Debug + objekt::Clone {
    fn packed_a_len(&self) -> usize;
    fn packed_a_alignment(&self) -> usize;
    fn pack_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize);
    fn b_pack(&self) -> PackB<T>;

    fn mat_mul_prepacked(&self, pa: *const T, pb: *const T, c: *mut T, rsc: isize, csc: isize);

    fn m(&self) -> usize;
    fn n(&self) -> usize;
    fn k(&self) -> usize;
}

clone_trait_object!(<T> MatMul<T> where T: Copy + Add + Mul + Zero);

pub trait PackedMatMulKer<T: Copy + Add + Mul + Zero>: Copy + Clone + Debug + Send + Sync {
    #[inline(always)]
    fn name() -> &'static str;
    #[inline(always)]
    fn kernel(k: usize, a: *const T, b: *const T, c: *mut T, rsc: usize, csc: usize);
    #[inline(always)]
    fn mr() -> usize;
    #[inline(always)]
    fn nr() -> usize;
    #[inline(always)]
    fn alignment_bytes_a() -> usize;
    #[inline(always)]
    fn alignment_bytes_b() -> usize;
}

#[derive(Copy, Clone)]
pub struct PackedMatMul<K, T>
where
    K: PackedMatMulKer<T> + Debug,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    m: usize,
    k: usize,
    n: usize,
    _kernel: PhantomData<(K, T)>,
}

impl<K, T> std::fmt::Debug for PackedMatMul<K, T>
where
    K: PackedMatMulKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            fmt,
            "MM m:{} k:{} n:{} {}({}x{})",
            self.m,
            self.k,
            self.n,
            K::name(),
            K::mr(),
            K::nr()
        )
    }
}

impl<K, T> PackedMatMul<K, T>
where
    K: PackedMatMulKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    pub fn new(m: usize, k: usize, n: usize) -> PackedMatMul<K, T> {
        PackedMatMul { m, k, n, _kernel: PhantomData }
    }

    fn pack_panel_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize, rows: usize) {
        let mr = K::mr();
        for i in 0..self.k {
            for j in 0..rows {
                unsafe {
                    *pa.offset((i * mr + j) as isize) =
                        *a.offset(i as isize * csa + j as isize * rsa)
                }
            }
        }
    }
}

impl<K, T> MatMul<T> for PackedMatMul<K, T>
where
    K: PackedMatMulKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync + PartialEq,
{
    fn packed_a_alignment(&self) -> usize {
        K::alignment_bytes_a()
    }
    fn packed_a_len(&self) -> usize {
        let mr = K::mr();
        (self.m + mr - 1) / mr * mr * self.k
    }

    fn pack_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize) {
        let mr = K::mr();
        assert!(pa as usize % K::alignment_bytes_a() == 0);
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
            assert_eq!(*pa, *a);
        }
    }

    fn b_pack(&self) -> PackB<T> {
        PackB::new(self.k, self.n, K::nr(), K::alignment_bytes_b())
    }

    fn mat_mul_prepacked(&self, pa: *const T, pb: *const T, c: *mut T, rsc: isize, csc: isize) {
        assert!(pa as usize % K::alignment_bytes_a() == 0);
        assert!(pb as usize % K::alignment_bytes_b() == 0);
        let mr = K::mr();
        let nr = K::nr();
        let m = self.m;
        let k = self.k;
        let n = self.n;
        let mut tmpc = vec![T::zero(); mr * nr];
        unsafe {
            for ia in 0..m / mr {
                for ib in 0..n / nr {
                    K::kernel(
                        k,
                        pa.offset((ia * k * mr) as isize),
                        pb.offset((ib * k * nr) as isize),
                        c.offset((mr * ia) as isize * rsc + (nr * ib) as isize * csc),
                        rsc as usize,
                        csc as usize,
                    );
                }
                if n % nr != 0 {
                    K::kernel(
                        k,
                        pa.offset((ia * k * mr) as isize),
                        pb.offset((n / nr * k * nr) as isize),
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
            if m % mr != 0 {
                for ib in 0..n / nr {
                    K::kernel(
                        k,
                        pa.offset((m / mr * mr * k) as isize),
                        pb.offset((ib * nr * k) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                        1,
                    );
                    for y in 0..(m % mr) {
                        for x in 0..nr {
                            *c.offset(
                                (y + m / mr * mr) as isize * rsc + (x + ib * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
                if n % nr != 0 {
                    K::kernel(
                        k,
                        pa.offset((m / mr * mr * k) as isize),
                        pb.offset((n / nr * nr * k) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                        1,
                    );
                    for y in 0..(m % mr) {
                        for x in 0..(n % nr) {
                            *c.offset(
                                (y + m / mr * mr) as isize * rsc + (x + n / nr * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
            }
        }
    }

    fn m(&self) -> usize {
        self.m
    }

    fn k(&self) -> usize {
        self.k
    }

    fn n(&self) -> usize {
        self.n
    }
}

#[derive(Debug)]
pub struct PackedWriter<'p, T>
where
    T: Copy + Debug,
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
    T: Copy + Debug,
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

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::align;
    use proptest::prelude::*;
    use proptest::*;

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
            .prop_map(|(k, a, b)| {
                (
                    k,
                    align::realign_vec(a, MM::alignment_bytes_a()),
                    align::realign_vec(b, MM::alignment_bytes_b()),
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
                expect[x + y * MM::nr()] =
                    (0..k).map(|k| pa[k * MM::mr() + y] * pb[k * MM::nr() + x]).sum::<f32>();
            }
        }
        let mut found = vec![9999.0f32; expect.len()];
        MM::kernel(k, pa.as_ptr(), pb.as_ptr(), found.as_mut_ptr(), MM::nr(), 1);
        prop_assert_eq!(found, expect);
        Ok(())
    }

    pub fn strat_mat_mul() -> BoxedStrategy<(usize, usize, usize, Vec<f32>, Vec<f32>)> {
        (1usize..35, 1usize..35, 1usize..35)
            .prop_flat_map(move |(m, k, n)| {
                (
                    Just(m),
                    Just(k),
                    Just(n),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), m * k),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), n * k),
                )
            })
            .boxed()
    }

    pub fn test_mat_mul_prep_f32<MM: MatMul<f32>>(
        mm: MM,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        unsafe {
            let mut packed_a: Vec<f32> =
                align::uninitialized(mm.packed_a_len(), mm.packed_a_alignment());
            mm.pack_a(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

            let mut packed_b: Vec<f32> =
                align::uninitialized(mm.b_pack().len(), mm.b_pack().alignment());
            mm.b_pack().pack(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

            let mut found = vec![9999.0f32; m * n];

            mm.mat_mul_prepacked(
                packed_a.as_ptr(),
                packed_b.as_ptr(),
                found.as_mut_ptr(),
                n as isize,
                1,
            );
            let mut expect = vec![0.0f32; m * n];
            for x in 0..n {
                for y in 0..m {
                    for i in 0..k {
                        expect[x + y * n] += a[i + k * y] * b[x + i * n]
                    }
                }
            }
            prop_assert_eq!(found, expect);
        }
        Ok(())
    }

}
