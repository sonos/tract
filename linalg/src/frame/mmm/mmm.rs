use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use num_traits::Zero;

use crate::frame::{PackA, PackB};

use super::*;

#[derive(Default)]
struct ScratchSpace<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync + Default,
{
    uspecs: Vec<FusedKerSpec<T>>,
    non_linear_buffers: Vec<Vec<T>>,
}

impl<T> ScratchSpace<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync + Default,
{
    unsafe fn non_linear<K: MatMatMulKer<T>>(
        &mut self,
        specs: &[FusedSpec<T>],
        down: usize,
        right: usize,
    ) -> *const FusedKerSpec<T> {
        self.uspecs.clear();
        for spec in specs {
            let s = match spec {
                FusedSpec::Min(m) => FusedKerSpec::Min(*m),
                FusedSpec::Max(m) => FusedKerSpec::Max(*m),
                FusedSpec::AddC => FusedKerSpec::AddC,
                FusedSpec::PerRowMul(v) => {
                    let have = v.len() - down * K::mr();
                    let ptr = if have < K::mr() {
                        let mut buf = vec![T::zero(); K::mr()];
                        buf[..have].copy_from_slice(&v[down * K::mr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr().add(down * K::mr())
                    };
                    FusedKerSpec::PerRowMul(ptr)
                }
                FusedSpec::PerRowAdd(v) => {
                    let have = v.len() - down * K::mr();
                    let ptr = if have < K::mr() {
                        let mut buf = vec![T::zero(); K::mr()];
                        buf[..have].copy_from_slice(&v[down * K::mr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr().add(down * K::mr())
                    };
                    FusedKerSpec::PerRowAdd(ptr)
                }
                FusedSpec::PerColMul(v) => {
                    let have = v.len() - right * K::nr();
                    let ptr = if have < K::nr() {
                        let mut buf = vec![T::zero(); K::nr()];
                        buf[..have].copy_from_slice(&v[right * K::nr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr().add(right * K::nr())
                    };
                    FusedKerSpec::PerColMul(ptr)
                }
                FusedSpec::PerColAdd(v) => {
                    let have = v.len() - right * K::nr();
                    let ptr = if have < K::nr() {
                        let mut buf = vec![T::zero(); K::nr()];
                        buf[..have].copy_from_slice(&v[right * K::nr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr().add(right * K::nr())
                    };
                    FusedKerSpec::PerColAdd(ptr)
                }
            };
            self.uspecs.push(s);
        }
        self.uspecs.push(FusedKerSpec::Done);
        self.uspecs.as_ptr()
    }
}

pub trait MatMatMul<T>: Send + Sync + Debug + fmt::Display + objekt::Clone
where
    T: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + Send + Sync,
{
    fn a_pack(&self) -> PackA<T>;
    fn b_pack(&self) -> PackB<T>;

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

    unsafe fn run(&self, a: *const T, b: *const T, c: *mut T, non_linear: &[FusedSpec<T>]);
}

clone_trait_object!(<T> MatMatMul<T> where T: Copy + Add + Mul + Zero);

#[derive(Debug, Clone)]
pub struct MatMatMulImpl<K, T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync,
    K: MatMatMulKer<T>,
{
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a_storage: MatrixStoreSpec,
    pub b_storage: MatrixStoreSpec,
    pub c_storage: MatrixStoreSpec,
    phantom: PhantomData<(K, T)>,
}

impl<K, T> MatMatMulImpl<K, T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync,
    K: MatMatMulKer<T>,
{
    pub fn new(m: usize, k: usize, n: usize) -> MatMatMulImpl<K, T> {
        MatMatMulImpl {
            m,
            k,
            n,
            a_storage: MatrixStoreSpec::Packed { panel_len: (k * K::mr()) },
            b_storage: MatrixStoreSpec::Packed { panel_len: (k * K::nr()) },
            c_storage: MatrixStoreSpec::Strides {
                row_byte_stride: (n * std::mem::size_of::<T>()) as isize,
                col_byte_stride: (std::mem::size_of::<T>()) as isize,
                mr: K::mr(),
                nr: K::nr(),
            },
            phantom: PhantomData,
        }
    }
}

impl<K, T> MatMatMul<T> for MatMatMulImpl<K, T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync + Default + fmt::Display,
    K: MatMatMulKer<T>,
{
    fn a_pack(&self) -> PackA<T> {
        PackA::new(self.k, self.m, K::mr(), K::alignment_bytes_packed_a())
    }

    fn b_pack(&self) -> PackB<T> {
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
            cols_offsets.iter().map(|o| o * std::mem::size_of::<T>() as isize).collect();
        while col_byte_offsets.len() < wanted {
            col_byte_offsets.push(*col_byte_offsets.last().unwrap());
        }
        // repeat the last offset four times to simplify kernel loop unrolling
        let mut row_byte_offsets: Vec<_> = Vec::with_capacity(rows_offsets.len() + 4);
        row_byte_offsets.set_len(rows_offsets.len() + 4);
        for i in 0..rows_offsets.len() {
            *row_byte_offsets.get_unchecked_mut(i) =
                *rows_offsets.get_unchecked(i) * std::mem::size_of::<T>() as isize;
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
            byte_stride: stride * std::mem::size_of::<T>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn b_vec_from_data(&mut self) {
        self.b_vec_from_data_and_stride(1)
    }

    unsafe fn c_from_data_and_strides(&mut self, row_stride: isize, col_stride: isize) {
        self.c_storage = MatrixStoreSpec::Strides {
            row_byte_stride: row_stride * std::mem::size_of::<T>() as isize,
            col_byte_stride: col_stride * std::mem::size_of::<T>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn c_vec_from_data_and_stride(&mut self, stride: isize) {
        self.c_storage = MatrixStoreSpec::VecStride {
            byte_stride: stride * std::mem::size_of::<T>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn c_vec_from_data(&mut self) {
        self.c_vec_from_data_and_stride(1)
    }

    unsafe fn run(&self, a: *const T, b: *const T, c: *mut T, non_linear: &[FusedSpec<T>]) {
        let mr = K::mr();
        let nr = K::nr();
        let m = self.m;
        let k = self.k;
        let n = self.n;
        let mut scratch = ScratchSpace::default();
        let tmpc = vec![T::zero(); mr * nr];
        let tmp_c_storage = MatrixStoreSpec::Strides {
            row_byte_stride: (std::mem::size_of::<T>() * nr) as isize,
            col_byte_stride: std::mem::size_of::<T>() as isize,
            mr,
            nr,
        };
        let ref mut tmp_tile = tmp_c_storage.wrap(tmpc.as_ptr());
        let ref linear = LinearSpec::Mul { k };
        let a = self.a_storage.wrap(a);
        let b = self.b_storage.wrap(b);
        let mut c = self.c_storage.wrap(c);
        for ia in 0..m / mr {
            let ref a = a.panel_a(ia);
            for ib in 0..n / nr {
                let ref b = b.panel_b(nr, ib, nr);
                let ref direct_c = c.tile_c(ia, ib);
                let non_linear = scratch.non_linear::<K>(non_linear, ia, ib);
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
                let non_linear = scratch.non_linear::<K>(non_linear, ia, n / nr);
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
                let non_linear = scratch.non_linear::<K>(non_linear, m / mr, ib);
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
                let non_linear = scratch.non_linear::<K>(non_linear, m / mr, n / nr);
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

impl<K, T> fmt::Display for MatMatMulImpl<K, T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync + fmt::Display,
    K: MatMatMulKer<T>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "A:{}, B:{} C:{} (m:{}, k:{}, n:{})", self.a_storage, self.b_storage, self.c_storage, self.m, self.k, self.n)
    }
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::align::Buffer;
    use proptest::prelude::*;

    #[macro_export]
    macro_rules! mmm_frame_tests {
        ($cond:expr, $ker:ty) => {
            mod frame {
                #[allow(unused_imports)]
                use crate::frame::mmm::mmm::test::*;
                proptest::proptest! {
                    #[test]
                    fn mat_mul_prepacked((m, k, n, ref a, ref b) in strat_mat_mat_mul()) {
                        if $cond {
                            test_mat_mat_mul_prep_f32::<$ker>(m, k, n, a, b)?
                        }
                    }

                    #[test]
                    fn mat_vec_prepacked((m, k, ref a, ref b) in strat_mat_vec_mul()) {
                        if $cond {
                            test_mat_vec_mul_prep_f32::<$ker>(m, k, a, b)?
                        }
                    }

                    #[test]
                    fn conv_prepacked(pb in strat_conv_1d()) {
                        if $cond {
                            crate::check_close(&*pb.run::<$ker>(), &*pb.expected())?;
                        }
                    }

                }

                #[test]
                fn mat_mul_1() {
                    if $cond {
                        test_mat_mat_mul_prep_f32::<$ker>(
                            3,
                            4,
                            2,
                            &[-3.0, 3.0, 5.0, -5.0, 6.0, 0.0, -6.0, -5.0, 0.0, 0.0, 9.0, 7.0],
                            &[-8.0, 5.0, 5.0, -3.0, 5.0, 7.0, -8.0, -1.0],
                        )
                        .unwrap()
                    }
                }

                #[test]
                fn mat_mul_1_2_1() {
                    if $cond {
                        test_mat_mat_mul_prep_f32::<$ker>(1, 2, 1, &[0.0, 1.0], &[0.0, 1.0])
                            .unwrap()
                    }
                }

                #[test]
                fn conv_prepacked_1() {
                    if $cond {
                        let filters = vec![1.0f32];
                        let data = vec![0.0f32, 1.0];
                        let pb = ConvProblem {
                            ci: 1,
                            co: 1,
                            kt: 1,
                            stride: 1,
                            dilation: 1,
                            filters,
                            data,
                        };
                        crate::check_close(&*pb.run::<$ker>(), &*pb.expected()).unwrap();
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
                        crate::check_close(&*pb.run::<$ker>(), &*pb.expected()).unwrap();
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

    pub fn test_mat_mat_mul_prep_f32<K: MatMatMulKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        let op = MatMatMulImpl::<K, f32>::new(m, k, n);
        unsafe {
            let mut packed_a = Buffer::uninitialized(op.a_pack().len(), op.a_pack().alignment());
            op.a_pack().pack(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

            let mut packed_b = Buffer::uninitialized(op.b_pack().len(), op.b_pack().alignment());
            op.b_pack().pack(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

            let mut found = vec![9999.0f32; m * n];

            op.run(packed_a.as_ptr(), packed_b.as_ptr(), found.as_mut_ptr(), &[]);

            let mut expected = vec![0.0f32; m * n];
            for x in 0..n {
                for y in 0..m {
                    for i in 0..k {
                        expected[x + y * n] += a[i + k * y] * b[x + i * n]
                    }
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

    pub fn test_mat_vec_mul_prep_f32<K: MatMatMulKer<f32>>(
        m: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        unsafe {
            let mut op = MatMatMulImpl::<K, f32>::new(m, k, 1);
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

    pub unsafe fn fused_op<K: MatMatMulKer<f32>, F: Fn(&mut [f32])>(
        m: usize,
        k: usize,
        n: usize,
        spec: &[FusedSpec<f32>],
        expect: F,
    ) -> proptest::test_runner::TestCaseResult {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];
        let op = MatMatMulImpl::<K, f32>::new(m, k, n);

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

    pub unsafe fn row_add<K: MatMatMulKer<f32>>(
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

    pub unsafe fn row_mul<K: MatMatMulKer<f32>>(
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

    pub unsafe fn col_add<K: MatMatMulKer<f32>>(
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

    pub unsafe fn col_mul<K: MatMatMulKer<f32>>(
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

    pub unsafe fn max<K: MatMatMulKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        fused_op::<K, _>(m, k, n, &[FusedSpec::Max(5f32)], |exp| {
            exp.iter_mut().for_each(|x| *x = x.max(5f32))
        })
    }

    pub unsafe fn min<K: MatMatMulKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        fused_op::<K, _>(m, k, n, &[FusedSpec::Min(1f32)], |exp| {
            exp.iter_mut().for_each(|x| *x = x.min(1f32))
        })
    }

    #[derive(Clone, Debug)]
    pub struct ConvProblem {
        pub ci: usize,
        pub co: usize,
        pub kt: usize,
        pub stride: usize,
        pub dilation: usize,
        pub filters: Vec<f32>,
        pub data: Vec<f32>,
    }

    impl ConvProblem {
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
        pub fn expected(&self) -> Vec<f32> {
            let mut expect = vec![0.0f32; self.co * self.output_width()];
            for x in 0..self.output_width() {
                for ico in 0..self.co {
                    for ikt in 0..self.kt {
                        for ici in 0..self.ci {
                            let f = self.filters[ici * self.kt + ikt + self.ci * self.kt * ico];
                            let d = self.data
                                [x * self.stride + ikt * self.dilation + ici * self.input_width()];
                            expect[x + ico * self.output_width()] += f * d;
                        }
                    }
                }
            }
            expect
        }

        pub fn run<K: MatMatMulKer<f32>>(&self) -> Vec<f32> {
            unsafe {
                let mut op = MatMatMulImpl::<K, f32>::new(self.m(), self.k(), self.n());
                op.b_from_data_and_offsets(&self.data_rows_offsets(), &self.data_cols_offsets());
                let mut packed_a =
                    Buffer::uninitialized(op.a_pack().len(), op.a_pack().alignment());
                op.a_pack().pack(
                    packed_a.as_mut_ptr(),
                    self.filters.as_ptr(),
                    self.k() as isize,
                    1,
                );

                let mut found = vec![9999.0f32; self.co * self.output_width()];
                op.run(packed_a.as_ptr(), self.data.as_ptr(), found.as_mut_ptr(), &[]);
                found
            }
        }
    }

    pub fn strat_conv_1d() -> BoxedStrategy<ConvProblem> {
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
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), ci * co * kt),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), t * ci),
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
