use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use num_traits::Zero;

use crate::frame::{PackA, PackB};

use super::fuse::ScratchSpaceFusedNonLinear;
use super::*;

pub trait MatMatMul<TA, TB, TC, TI>: Debug + fmt::Display + objekt::Clone + Send + Sync
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy + Debug,
    TI: Copy + Add + Mul + Zero + Debug,
{
    fn a_pack(&self) -> PackA<TA>;
    fn b_pack(&self) -> PackB<TB>;

    fn a_storage(&self) -> &MatrixStoreSpec;
    fn b_storage(&self) -> &MatrixStoreSpec;
    fn c_storage(&self) -> &MatrixStoreSpec;

    fn m(&self) -> usize;
    fn k(&self) -> usize;
    fn n(&self) -> usize;

    unsafe fn b_from_data_and_offsets(&mut self, rows_offsets: &[isize], cols_offsets: &[isize]);

    unsafe fn b_vec_from_data_and_stride(&mut self, stride: isize);
    unsafe fn b_vec_from_data(&mut self);

    unsafe fn c_from_data_and_strides(&mut self, row_stride: isize, col_stride: isize);

    unsafe fn c_vec_from_data_and_stride(&mut self, stride: isize);
    unsafe fn c_vec_from_data(&mut self);

    unsafe fn run(&self, a: *const TA, b: *const TB, c: *mut TC, non_linear: &[FusedSpec<TI>]);
}

clone_trait_object!(<TA, TB, TC, TI> MatMatMul<TA, TB, TC, TI> where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy + Debug,
    TI: Copy + Add + Mul + Zero + Debug,
);

#[derive(Debug, Clone)]
pub struct MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy + Debug,
    TI: Copy + Add + Mul + Zero + Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    pub m: usize,
    pub k: usize,
    pub n: usize,

    pub a_storage: MatrixStoreSpec,
    pub b_storage: MatrixStoreSpec,
    pub c_storage: MatrixStoreSpec,

    phantom: PhantomData<(K, TA, TB, TC, TI)>,
}

unsafe impl<K, TA, TB, TC, TI> Send for MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy + Debug,
    TI: Copy + Add + Mul + Zero + Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
}

unsafe impl<K, TA, TB, TC, TI> Sync for MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy + Debug,
    TI: Copy + Add + Mul + Zero + Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
}

impl<K, TA, TB, TC, TI> MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy + Debug,
    TI: Copy + Add + Mul + Zero + Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    pub fn new(m: usize, k: usize, n: usize) -> MatMatMulImpl<K, TA, TB, TC, TI> {
        MatMatMulImpl {
            m,
            k,
            n,
            a_storage: MatrixStoreSpec::Packed { panel_len: (k * K::mr()) },
            b_storage: MatrixStoreSpec::Packed { panel_len: (k * K::nr()) },
            c_storage: MatrixStoreSpec::Strides {
                row_byte_stride: (n * std::mem::size_of::<TC>()) as isize,
                col_byte_stride: (std::mem::size_of::<TC>()) as isize,
                mr: K::mr(),
                nr: K::nr(),
            },
            phantom: PhantomData,
        }
    }
}

impl<K, TA, TB, TC, TI> MatMatMul<TA, TB, TC, TI> for MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + Debug,
    TB: Copy + Zero + Debug,
    TC: Copy + Debug,
    TI: Copy + Add + Mul + Zero + Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn a_pack(&self) -> PackA<TA> {
        PackA::new(self.k, self.m, K::mr(), K::alignment_bytes_packed_a())
    }

    fn b_pack(&self) -> PackB<TB> {
        PackB::new(self.k, self.n, K::nr(), K::alignment_bytes_packed_b())
    }

    fn m(&self) -> usize {
        self.m
    }

    fn n(&self) -> usize {
        self.n
    }

    fn k(&self) -> usize {
        self.k
    }

    fn a_storage(&self) -> &MatrixStoreSpec {
        &self.a_storage
    }
    fn b_storage(&self) -> &MatrixStoreSpec {
        &self.b_storage
    }
    fn c_storage(&self) -> &MatrixStoreSpec {
        &self.c_storage
    }

    unsafe fn b_from_data_and_offsets(&mut self, rows_offsets: &[isize], cols_offsets: &[isize]) {
        debug_assert!(rows_offsets.len() > 0);
        debug_assert!(cols_offsets.len() > 0);
        // repeat the last offset to get to the panel boundary (pad to next multiple of nr)
        let wanted = (cols_offsets.len() + K::nr() - 1) / K::nr() * K::nr();
        let mut col_byte_offsets: Vec<_> =
            cols_offsets.iter().map(|o| o * std::mem::size_of::<TB>() as isize).collect();
        while col_byte_offsets.len() < wanted {
            col_byte_offsets.push(*col_byte_offsets.last().unwrap());
        }
        // repeat the last offset four times to simplify kernel loop unrolling
        let mut row_byte_offsets: Vec<_> = Vec::with_capacity(rows_offsets.len() + 4);
        row_byte_offsets.set_len(rows_offsets.len() + 4);
        for i in 0..rows_offsets.len() {
            *row_byte_offsets.get_unchecked_mut(i) =
                *rows_offsets.get_unchecked(i) * std::mem::size_of::<TB>() as isize;
        }
        let pad = *row_byte_offsets.get_unchecked(rows_offsets.len() - 1);
        for i in 0..4 {
            *row_byte_offsets.get_unchecked_mut(rows_offsets.len() + i) = pad;
        }
        self.b_storage =
            MatrixStoreSpec::OffsetsAndPtrs { col_byte_offsets, row_byte_offsets, nr: K::nr() };
    }

    unsafe fn b_vec_from_data_and_stride(&mut self, stride: isize) {
        self.b_storage = MatrixStoreSpec::VecStride {
            byte_stride: stride * std::mem::size_of::<TB>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn b_vec_from_data(&mut self) {
        self.b_vec_from_data_and_stride(1)
    }

    unsafe fn c_from_data_and_strides(&mut self, row_stride: isize, col_stride: isize) {
        self.c_storage = MatrixStoreSpec::Strides {
            row_byte_stride: row_stride * std::mem::size_of::<TC>() as isize,
            col_byte_stride: col_stride * std::mem::size_of::<TC>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn c_vec_from_data_and_stride(&mut self, stride: isize) {
        self.c_storage = MatrixStoreSpec::VecStride {
            byte_stride: stride * std::mem::size_of::<TC>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn c_vec_from_data(&mut self) {
        self.c_vec_from_data_and_stride(1)
    }

    unsafe fn run(&self, a: *const TA, b: *const TB, c: *mut TC, non_linear: &[FusedSpec<TI>]) {
        let mr = K::mr();
        let nr = K::nr();
        let m = self.m;
        let n = self.n;
        let mut scratch = ScratchSpaceFusedNonLinear::default();
        let mut tmpc = Vec::with_capacity(mr * nr);
        tmpc.set_len(mr * nr);
        let tmp_c_storage = MatrixStoreSpec::Strides {
            row_byte_stride: (std::mem::size_of::<TC>() * nr) as isize,
            col_byte_stride: std::mem::size_of::<TC>() as isize,
            mr,
            nr,
        };
        let ref mut tmp_tile = tmp_c_storage.wrap(tmpc.as_ptr());
        let a = self.a_storage.wrap(a);
        let b = self.b_storage.wrap(b);
        let mut c = self.c_storage.wrap(c);
        let ref linear = LinearSpec::k(self.k);
        for ia in 0..m / mr {
            let ref a = a.panel_a(ia);
            for ib in 0..n / nr {
                let ref b = b.panel_b(nr, ib, nr);
                let ref direct_c = c.tile_c(ia, ib);
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(non_linear, ia, ib);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: a as _,
                    b: b as _,
                    c: direct_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
            }
            if n % nr != 0 {
                let ref b = b.panel_b(nr, n / nr, n % nr);
                let ref tmp_tile_c = tmp_tile.tile_c(0, 0);
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(non_linear, ia, n / nr);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile(ia, n / nr, mr, n % nr, &*tmpc);
            }
        }
        if m % mr != 0 {
            let ref panel_a = a.panel_a(m / mr);
            let ref tmp_tile_c = tmp_tile.tile_c(0, 0);
            for ib in 0..n / nr {
                let ref b = b.panel_b(nr, ib, nr);
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(non_linear, m / mr, ib);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: panel_a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile(m / mr, ib, m % mr, nr, &*tmpc);
            }
            if n % nr != 0 {
                let ref b = b.panel_b(nr, n / nr, n % nr);
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(non_linear, m / mr, n / nr);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: panel_a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile(m / mr, n / nr, m % mr, n % nr, &*tmpc);
            }
        }
    }
}

impl<K, TA, TB, TC, TI> fmt::Display for MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy + Debug,
    TI: Copy + Add + Mul + Zero + Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "A:{}, B:{} C:{} (m:{}, k:{}, n:{})",
            self.a_storage, self.b_storage, self.c_storage, self.m, self.k, self.n
        )
    }
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::align::Buffer;
    use num_traits::{AsPrimitive, Bounded};
    use proptest::prelude::*;

    #[macro_export]
    macro_rules! mmm_frame_tests {
        ($cond:expr, $ker:ty, $ta:ty, $tb:ty, $tc:ty, $ti:ty) => {
            mod frame {
                #[allow(unused_imports)]
                use $crate::frame::mmm::mmm::test::*;
                use $crate::num_traits::AsPrimitive;

                proptest::proptest! {
                    #[test]
                    fn mat_mul_prepacked((m, k, n, ref a, ref b) in strat_mat_mat_mul()) {
                        if $cond {
                            test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(m, k, n, a, b)?
                        }
                    }

                    #[test]
                    fn mat_vec_prepacked((m, k, ref a, ref b) in strat_mat_vec_mul()) {
                        if $cond {
                            test_mat_vec_mul_prep::<$ker>(m, k, a, b)?
                        }
                    }

                    #[test]
                    fn conv_prepacked(pb in strat_conv_1d()) {
                        if $cond {
                            crate::check_close(&*pb.run::<$ker, $ta, $tb>(), &*pb.expected())?;
                        }
                    }

                }

                #[test]
                fn mat_mul_1() {
                    if $cond {
                        let a: Vec<$ta> = [-3isize, 3, 5, -5, 6, 0, -6, -5, 0, 0, 9, 7]
                            .iter()
                            .map(|x| x.as_())
                            .collect();
                        let b: Vec<$tb> =
                            [-8isize, 5, 5, -3, 5, 7, -8, -1].iter().map(|x| x.as_()).collect();
                        test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(3, 4, 2, &*a, &*b)
                            .unwrap()
                    }
                }

                #[test]
                fn mat_mul_1_2_1() {
                    if $cond {
                        test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(
                            1,
                            2,
                            1,
                            &[0.0f32.as_(), 1.0],
                            &[0.0, 1.0],
                        )
                        .unwrap()
                    }
                }

                #[test]
                fn conv_prepacked_1() {
                    if $cond {
                        let filters: Vec<$ta> = vec![1isize.as_()];
                        let data: Vec<$tb> = vec![0.as_(), 1.as_()];
                        let pb = ConvProblem {
                            ci: 1,
                            co: 1,
                            kt: 1,
                            stride: 1,
                            dilation: 1,
                            filters,
                            data,
                        };
                        crate::check_close(&*pb.run::<$ker, $ta, $tb>(), &*pb.expected()).unwrap();
                    }
                }

                #[test]
                fn conv_prepacked_2() {
                    if $cond {
                        let mut filters = vec![0.0f32; 3 * 14 * 2];
                        filters[13 * 6 + 5] = 1.0;
                        let mut data = vec![0.0f32; 3 * 10];
                        data[8 + 2 * 10] = 1.0; // last used input
                        let pb = ConvProblem {
                            ci: 3,
                            co: 14,
                            kt: 2,
                            stride: 3,
                            dilation: 2,
                            filters,
                            data,
                        };
                        crate::check_close(&*pb.run::<$ker, f32, f32>(), &*pb.expected()).unwrap();
                    }
                }

                #[test]
                fn row_mul_2_1_3() {
                    if $cond {
                        unsafe { row_mul::<$ker>(2, 1, 3).unwrap() }
                    }
                }

                #[test]
                fn row_add_2_1_3() {
                    if $cond {
                        unsafe { row_add::<$ker>(2, 1, 3).unwrap() }
                    }
                }

                #[test]
                fn col_mul_2_1_3() {
                    if $cond {
                        unsafe { col_mul::<$ker>(2, 1, 3).unwrap() }
                    }
                }

                #[test]
                fn col_add_2_1_3() {
                    if $cond {
                        unsafe { col_add::<$ker>(2, 1, 3).unwrap() }
                    }
                }

                #[test]
                fn max_2_1_3() {
                    if $cond {
                        unsafe { max::<$ker>(2, 1, 3).unwrap() }
                    }
                }

                #[test]
                fn min_2_1_3() {
                    if $cond {
                        unsafe { min::<$ker>(2, 3, 3).unwrap() }
                    }
                }
            }
        };
    }

    pub fn strat_mat_mat_mul() -> BoxedStrategy<(usize, usize, usize, Vec<f32>, Vec<f32>)> {
        (1usize..5, 1usize..5, 1usize..5)
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

    pub fn strat_mat_vec_mul() -> BoxedStrategy<(usize, usize, Vec<f32>, Vec<f32>)> {
        (1usize..5, 1usize..5)
            .prop_flat_map(move |(m, k)| {
                (
                    Just(m),
                    Just(k),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), m * k),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), k),
                )
            })
            .boxed()
    }

    pub fn test_mat_mat_mul_prep<K: MatMatMulKer<TA, TB, TC, TI>, TA, TB, TC, TI>(
        m: usize,
        k: usize,
        n: usize,
        a: &[TA],
        b: &[TB],
    ) -> Result<(), proptest::test_runner::TestCaseError>
    where
        TA: Copy + Zero + Debug + AsPrimitive<f64> + AsPrimitive<TI>,
        TB: Copy + Zero + Debug + AsPrimitive<f64> + AsPrimitive<TI>,
        TC: Copy + Zero + Debug + fmt::Display + PartialEq + Bounded + 'static + AsPrimitive<f32>,
        TI: Copy + Add + Mul<Output = TI> + Zero + Debug + fmt::Display + 'static + AsPrimitive<TC>,
    {
        let op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(m, k, n);
        unsafe {
            let mut packed_a = Buffer::uninitialized(op.a_pack().len(), op.a_pack().alignment());
            op.a_pack().pack(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

            let mut packed_b = Buffer::uninitialized(op.b_pack().len(), op.b_pack().alignment());
            op.b_pack().pack(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

            let mut found = vec![TC::max_value(); m * n];

            op.run(packed_a.as_ptr(), packed_b.as_ptr(), found.as_mut_ptr(), &[]);

            let mut expected = vec![TC::zero(); m * n];
            for x in 0..n {
                for y in 0..m {
                    let mut v: TI = TI::zero();
                    for i in 0..k {
                        let a: TI = a[i + k * y].as_();
                        let b: TI = b[x + i * n].as_();
                        v = v + a * b;
                    }
                    expected[x + y * n] = v.as_();
                }
            }

            proptest::prop_assert!(
                found.iter().zip(expected.iter()).all(|(&a, &b)| (a.as_() - b.as_()).abs() < 0.001),
                "found: {:?} expected: {:?}",
                found,
                expected
            );
        }
        Ok(())
    }

    pub fn test_mat_vec_mul_prep<K: MatMatMulKer<f32, f32, f32, f32>>(
        m: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        unsafe {
            let mut op = MatMatMulImpl::<K, f32, f32, f32, f32>::new(m, k, 1);
            op.b_vec_from_data_and_stride(1);
            op.c_vec_from_data_and_stride(1);
            let mut packed_a = Buffer::uninitialized(op.a_pack().len(), op.a_pack().alignment());
            op.a_pack().pack(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

            let mut found = vec![9999.0f32; m];

            op.run(packed_a.as_ptr(), b.as_ptr(), found.as_mut_ptr(), &[]);

            let mut expected = vec![0.0f32; m];
            for y in 0..m {
                for i in 0..k {
                    expected[y] += a[i + k * y] * b[i]
                }
            }

            proptest::prop_assert!(
                found.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 0.001),
                "found: {:?} expected: {:?}",
                found,
                expected
            );
        }
        Ok(())
    }

    pub unsafe fn fused_op<K: MatMatMulKer<f32, f32, f32, f32>, F: Fn(&mut [f32])>(
        m: usize,
        k: usize,
        n: usize,
        spec: &[FusedSpec<f32>],
        expect: F,
    ) -> proptest::test_runner::TestCaseResult {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];
        let op = MatMatMulImpl::<K, f32, f32, f32, f32>::new(m, k, n);

        let mut packed_a = Buffer::uninitialized(op.a_pack().len(), op.a_pack().alignment());
        op.a_pack().pack(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

        let mut packed_b = Buffer::uninitialized(op.b_pack().len(), op.b_pack().alignment());
        op.b_pack().pack(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

        let mut found = vec![9999.0f32; m * n];

        op.run(packed_a.as_ptr(), packed_b.as_ptr(), found.as_mut_ptr(), spec);

        let mut expected = vec![0.0f32; m * n];
        for x in 0..n {
            for y in 0..m {
                for i in 0..k {
                    expected[x + y * n] += a[i + k * y] * b[x + i * n]
                }
            }
        }
        expect(&mut expected);

        proptest::prop_assert!(
            found.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 0.001),
            "found: {:?} expected: {:?}",
            found,
            expected
        );
        Ok(())
    }

    pub unsafe fn row_add<K: MatMatMulKer<f32, f32, f32, f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        let bias = (0..m).map(|f| f as f32).collect::<Vec<f32>>();
        fused_op::<K, _>(m, k, n, &[FusedSpec::PerRowAdd(bias.clone())], |exp| {
            for x in 0..n {
                for y in 0..m {
                    exp[x + y * n] += bias[y]
                }
            }
        })
    }

    pub unsafe fn row_mul<K: MatMatMulKer<f32, f32, f32, f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        let bias = (0..m).map(|f| f as f32).collect::<Vec<f32>>();
        fused_op::<K, _>(m, k, n, &[FusedSpec::PerRowMul(bias.clone())], |exp| {
            for x in 0..n {
                for y in 0..m {
                    exp[x + y * n] *= bias[y]
                }
            }
        })
    }

    pub unsafe fn col_add<K: MatMatMulKer<f32, f32, f32, f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        let bias = (0..n).map(|f| f as f32).collect::<Vec<f32>>();
        fused_op::<K, _>(m, k, n, &[FusedSpec::PerColAdd(bias.clone())], |exp| {
            for x in 0..n {
                for y in 0..m {
                    exp[x + y * n] += bias[x]
                }
            }
        })
    }

    pub unsafe fn col_mul<K: MatMatMulKer<f32, f32, f32, f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        let bias = (0..n).map(|f| f as f32).collect::<Vec<f32>>();
        fused_op::<K, _>(m, k, n, &[FusedSpec::PerColMul(bias.clone())], |exp| {
            for x in 0..n {
                for y in 0..m {
                    exp[x + y * n] *= bias[x]
                }
            }
        })
    }

    pub unsafe fn max<K: MatMatMulKer<f32, f32, f32, f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        fused_op::<K, _>(m, k, n, &[FusedSpec::Max(5f32)], |exp| {
            exp.iter_mut().for_each(|x| *x = x.max(5f32))
        })
    }

    pub unsafe fn min<K: MatMatMulKer<f32, f32, f32, f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        fused_op::<K, _>(m, k, n, &[FusedSpec::Min(1f32)], |exp| {
            exp.iter_mut().for_each(|x| *x = x.min(1f32))
        })
    }

    #[derive(Clone, Debug)]
    pub struct ConvProblem<TA, TB> {
        pub ci: usize,
        pub co: usize,
        pub kt: usize,
        pub stride: usize,
        pub dilation: usize,
        pub filters: Vec<TA>,
        pub data: Vec<TB>,
    }

    impl<TA, TB> ConvProblem<TA, TB>
    where
        TA: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq,
        TB: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq,
    {
        pub fn kernel_field(&self) -> usize {
            self.dilation * (self.kt - 1) + 1
        }
        // this is not n, but the T in NTC of input to direct convolution
        pub fn input_width(&self) -> usize {
            assert!(self.data.len() % self.ci == 0);
            self.data.len() / self.ci
        }
        pub fn output_width(&self) -> usize {
            (self.input_width() - self.kernel_field()) / self.stride + 1
        }
        pub fn m(&self) -> usize {
            self.co
        }
        pub fn k(&self) -> usize {
            self.ci * self.kt
        }
        pub fn n(&self) -> usize {
            self.output_width()
        }
        pub fn data_cols_offsets(&self) -> Vec<isize> {
            (0..self.output_width()).map(|i| (i * self.stride) as isize).collect()
        }
        pub fn data_rows_offsets(&self) -> Vec<isize> {
            (0..self.ci)
                .flat_map(move |ici| {
                    (0..self.kt)
                        .map(move |ikt| (ikt * self.dilation + ici * self.input_width()) as isize)
                })
                .collect()
        }

        pub fn expected<TC>(&self) -> Vec<TC>
        where
            TC: Copy + Add + Mul<Output = TC> + Zero + Debug + fmt::Display + PartialEq + 'static,
            TA: AsPrimitive<TC>,
            TB: AsPrimitive<TC>,
        {
            let mut expect = vec![TC::zero(); self.co * self.output_width()];
            for x in 0..self.output_width() {
                for ico in 0..self.co {
                    for ikt in 0..self.kt {
                        for ici in 0..self.ci {
                            let f = self.filters[ici * self.kt + ikt + self.ci * self.kt * ico];
                            let d = self.data
                                [x * self.stride + ikt * self.dilation + ici * self.input_width()];
                            let ref mut pv = expect[x + ico * self.output_width()];
                            *pv = *pv + f.as_() * d.as_();
                        }
                    }
                }
            }
            expect
        }

        pub fn run<K: MatMatMulKer<TA, TB, TC, TI>, TC, TI>(&self) -> Vec<TC>
        where
            TI: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + Default,
            TC: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + Bounded,
        {
            unsafe {
                let mut op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(self.m(), self.k(), self.n());
                op.b_from_data_and_offsets(&self.data_rows_offsets(), &self.data_cols_offsets());
                let mut packed_a =
                    Buffer::uninitialized(op.a_pack().len(), op.a_pack().alignment());
                op.a_pack().pack(
                    packed_a.as_mut_ptr(),
                    self.filters.as_ptr(),
                    self.k() as isize,
                    1,
                );

                let mut found: Vec<TC> = vec![TC::max_value(); self.co * self.output_width()];
                op.run(packed_a.as_ptr(), self.data.as_ptr(), found.as_mut_ptr(), &[]);
                found
            }
        }
    }

    pub fn strat_conv_1d<TA, TB>() -> BoxedStrategy<ConvProblem<TA, TB>>
    where
        TA: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + 'static,
        TB: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + 'static,
        isize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        (1usize..40, 1usize..40, 1usize..10, 1usize..5, 1usize..5)
            .prop_flat_map(|(ci, co, kt, stride, dilation)| {
                let min = ((kt - 1) * dilation + 1) * stride;
                (Just(ci), Just(co), Just(kt), Just(stride), Just(dilation), min..min + 10)
            })
            .prop_flat_map(move |(ci, co, kt, stride, dilation, t)| {
                (
                    Just(ci),
                    Just(co),
                    Just(kt),
                    Just(stride),
                    Just(dilation),
                    proptest::collection::vec((-10isize..10).prop_map(|a| a.as_()), ci * co * kt),
                    proptest::collection::vec((-10isize..10).prop_map(|a| a.as_()), t * ci),
                )
            })
            .prop_map(move |(ci, co, kt, stride, dilation, filters, data)| ConvProblem {
                ci,
                co,
                kt,
                stride,
                dilation,
                filters,
                data,
            })
            .boxed()
    }
}
