use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, Mul};

use super::PackB;
use std::marker::PhantomData;

pub trait VecMatMul<T: Copy + Add + Mul + Zero + Debug>:
    Send + Sync + Debug + objekt::Clone
{
    fn packed_a_len(&self) -> usize;
    fn packed_a_alignment(&self) -> usize;
    fn pack_a(&self, pa: *mut T, a: *const T, sa: isize);

    fn b_pack(&self) -> PackB<T>;

    fn vec_mat_mul_prepacked(&self, pa: *const T, pb: *const T, py: *mut T, sy: isize);

    fn n(&self) -> usize;
    fn k(&self) -> usize;
}

clone_trait_object!(<T> VecMatMul<T> where T: Copy + Add + Mul + Zero);

pub trait VecMatMulKer<T: Copy + Add + Mul + Zero>: Copy + Clone + Debug + Send + Sync {
    #[inline(always)]
    fn name() -> &'static str;
    #[inline(always)]
    fn kernel(k: usize, a: *const T, b: *const T, c: *mut T, sy: usize);
    #[inline(always)]
    fn nr() -> usize;
    #[inline(always)]
    fn alignment_bytes_a() -> usize;
    #[inline(always)]
    fn alignment_bytes_b() -> usize;
}

#[derive(Copy, Clone)]
pub struct PackedVecMatMul<K, T>
where
    K: VecMatMulKer<T> + Debug,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    k: usize,
    n: usize,
    _kernel: PhantomData<(K, T)>,
}

impl<K, T> std::fmt::Debug for PackedVecMatMul<K, T>
where
    K: VecMatMulKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "VMM m:1 k:{} n:{} {}({})", self.k, self.n, K::name(), K::nr())
    }
}

impl<K, T> PackedVecMatMul<K, T>
where
    K: VecMatMulKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    pub fn new(k: usize, n: usize) -> PackedVecMatMul<K, T> {
        PackedVecMatMul { k, n, _kernel: PhantomData }
    }
}

impl<K, T> VecMatMul<T> for PackedVecMatMul<K, T>
where
    K: VecMatMulKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync + PartialEq,
{
    fn packed_a_alignment(&self) -> usize {
        K::alignment_bytes_a()
    }

    fn packed_a_len(&self) -> usize {
        self.k
    }

    fn pack_a(&self, pa: *mut T, a: *const T, sa: isize) {
        for i in 0..(self.k as isize) {
            unsafe { *pa.offset(i) = *a.offset(i * sa) }
        }
    }

    fn b_pack(&self) -> PackB<T> {
        PackB::new(self.k, self.n, K::nr(), K::alignment_bytes_b())
    }

    fn vec_mat_mul_prepacked(&self, pa: *const T, pb: *const T, y: *mut T, sy: isize) {
        assert!(pa as usize % K::alignment_bytes_a() == 0);
        assert!(pb as usize % K::alignment_bytes_b() == 0);
        let nr = K::nr();
        let k = self.k;
        let n = self.n;
        let mut tmpc = vec![T::zero(); nr];
        unsafe {
            for i in 0..n / nr {
                K::kernel(
                    k,
                    pa,
                    pb.offset((i * k * nr) as isize),
                    y.offset((nr * i) as isize * sy),
                    sy as usize,
                );
            }
            if n % nr != 0 {
                K::kernel(k, pa, pb.offset((n / nr * k * nr) as isize), tmpc.as_mut_ptr(), 1);
                for x in 0..(n % nr) {
                    *y.offset((x + n / nr * nr) as isize * sy) = tmpc[x];
                }
            }
        }
    }

    fn k(&self) -> usize {
        self.k
    }

    fn n(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::align::Buffer;
    use proptest::prelude::*;

    pub fn strat_vec_mat_mul() -> BoxedStrategy<(usize, usize, Vec<f32>, Vec<f32>)> {
        (1usize..35, 1usize..35)
            .prop_flat_map(move |(k, n)| {
                (
                    Just(k),
                    Just(n),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), k),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), n * k),
                )
            })
            .boxed()
    }

    pub fn test_vec_mat_mul_prep_f32<VMM: VecMatMul<f32>>(
        mm: VMM,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        let mut packed_a = Buffer::uninitialized(mm.packed_a_len(), mm.packed_a_alignment());
        mm.pack_a(packed_a.as_mut_ptr(), a.as_ptr(), 1);

        let mut packed_b = Buffer::uninitialized(mm.b_pack().len(), mm.b_pack().alignment());
        mm.b_pack().pack(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

        let mut found = vec![9999.0f32; n];

        mm.vec_mat_mul_prepacked(packed_a.as_ptr(), packed_b.as_ptr(), found.as_mut_ptr(), 1);
        let mut expect = vec![0.0f32; n];
        for x in 0..n {
            for i in 0..k {
                expect[x] += a[i] * b[x + i * n]
            }
        }
        prop_assert_eq!(found, expect);
        Ok(())
    }
}
