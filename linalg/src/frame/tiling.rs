use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use num_traits::Zero;

use super::PackA;
use super::PackB;

use super::*;

#[repr(C, usize)]
#[derive(PartialEq, Clone, Debug)]
pub enum StorageSpec<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync,
{
    Strides { ptr: *const T, row_byte_stride: isize, col_byte_stride: isize, mr: usize, nr: usize },
    Packed { ptr: *const T, panel_len: usize },
    OffsetsAndPtrs { row_byte_offsets: Vec<isize>, col_ptrs: Vec<*const T>, nr: usize },
}

impl<T> StorageSpec<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync,
{
    unsafe fn panel_a(&self, i: usize) -> TileStorageSpec<T> {
        match self {
            StorageSpec::Packed { ptr, panel_len } => {
                TileStorageSpec::Packed { ptr: ptr.offset((panel_len * i) as isize) }
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn panel_b(&self, i: usize) -> TileStorageSpec<T> {
        match self {
            StorageSpec::Packed { ptr, panel_len } => {
                TileStorageSpec::Packed { ptr: ptr.offset((panel_len * i) as isize) }
            }
            StorageSpec::OffsetsAndPtrs { row_byte_offsets, col_ptrs, nr } => {
                TileStorageSpec::OffsetsAndPtrs {
                    row_byte_offsets: row_byte_offsets.as_ptr(),
                    col_ptrs: col_ptrs.as_ptr().offset((nr * i) as isize),
                }
            }
            _ => unimplemented!(),
        }
    }

    fn tile(&self, down: usize, right: usize) -> TileStorageSpec<T> {
        match self {
            StorageSpec::Strides { ptr, row_byte_stride, col_byte_stride, mr, nr } => {
                TileStorageSpec::Strides {
                    ptr: ((*ptr as isize)
                        + (*row_byte_stride as usize * down * mr
                            + *col_byte_stride as usize * right * nr)
                            as isize) as *mut T,
                    row_byte_stride: *row_byte_stride,
                    col_byte_stride: *col_byte_stride,
                }
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn set_from_tile(
        &mut self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &[T],
    ) {
        match self {
            StorageSpec::Strides { ptr, row_byte_stride, col_byte_stride, mr, nr } => {
                for y in 0..height {
                    for x in 0..width {
                        let ptr = ((*ptr as isize)
                            + (*row_byte_stride as usize * (down * *mr + y)
                                + *col_byte_stride as usize * (right * *nr + x))
                                as isize) as *mut T;
                        let value = *tile.get_unchecked(y * *nr + x);
                        *ptr = value;
                    }
                }
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(PartialEq, Clone)]
pub enum NonLinearSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Min(T),
    Max(T),
    AddC,
    PerRowMul(Vec<T>),
    PerRowAdd(Vec<T>),
    PerColMul(Vec<T>),
    PerColAdd(Vec<T>),
}

impl<T> Debug for NonLinearSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NonLinearSpec::Min(t) => write!(fmt, "Min({:?})", t),
            NonLinearSpec::Max(t) => write!(fmt, "Max({:?})", t),
            NonLinearSpec::AddC => write!(fmt, "AddC"),
            NonLinearSpec::PerRowMul(_) => write!(fmt, "PerRowMul"),
            NonLinearSpec::PerRowAdd(_) => write!(fmt, "PerRowAdd"),
            NonLinearSpec::PerColMul(_) => write!(fmt, "PerColMul"),
            NonLinearSpec::PerColAdd(_) => write!(fmt, "PerColAdd"),
        }
    }
}

#[derive(Default)]
struct ScratchSpace<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync + Default,
{
    uspecs: Vec<NonLinearUSpec<T>>,
    non_linear_buffers: Vec<Vec<T>>,
}

impl<T> ScratchSpace<T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync + Default,
{
    unsafe fn non_linear<K: TilingKer<T>>(
        &mut self,
        specs: &[NonLinearSpec<T>],
        down: usize,
        right: usize,
    ) -> *const NonLinearUSpec<T> {
        self.uspecs.clear();
        for spec in specs {
            let s = match spec {
                NonLinearSpec::Min(m) => NonLinearUSpec::Min(*m),
                NonLinearSpec::Max(m) => NonLinearUSpec::Max(*m),
                NonLinearSpec::AddC => NonLinearUSpec::AddC,
                NonLinearSpec::PerRowMul(v) => {
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
                    NonLinearUSpec::PerRowMul(ptr)
                }
                NonLinearSpec::PerRowAdd(v) => {
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
                    NonLinearUSpec::PerRowAdd(ptr)
                }
                NonLinearSpec::PerColMul(v) => {
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
                    NonLinearUSpec::PerColMul(ptr)
                }
                NonLinearSpec::PerColAdd(v) => {
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
                    NonLinearUSpec::PerColAdd(ptr)
                }
            };
            self.uspecs.push(s);
        }
        self.uspecs.push(NonLinearUSpec::Done);
        self.uspecs.as_ptr()
    }
}

pub trait Tile<T>: Send + Sync + Debug + objekt::Clone
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync,
{
    fn a_pack(&self) -> PackA<T>;
    fn b_pack(&self) -> PackB<T>;

    fn m(&self) -> usize;
    fn k(&self) -> usize;
    fn n(&self) -> usize;

    unsafe fn a_from_packed(&self, ptr: *const T) -> StorageSpec<T>;
    unsafe fn b_from_packed(&self, ptr: *const T) -> StorageSpec<T>;

    unsafe fn b_from_data_and_offsets(
        &self,
        data: *const T,
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> StorageSpec<T>;

    unsafe fn c_from_data_and_strides(
        &self,
        data: *const T,
        row_stride: isize,
        col_stride: isize,
    ) -> StorageSpec<T>;

    unsafe fn run(
        &self,
        a: &StorageSpec<T>,
        b: &StorageSpec<T>,
        c: &mut StorageSpec<T>,
        non_linear: &[NonLinearSpec<T>],
    );
}

clone_trait_object!(<T> Tile<T> where T: Copy + Add + Mul + Zero);

#[derive(Debug, Clone, new)]
pub struct TileOp<K, T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync,
    K: TilingKer<T>,
{
    pub m: usize,
    pub k: usize,
    pub n: usize,
    phantom: PhantomData<(K, T)>,
}

impl<K, T> Tile<T> for TileOp<K, T>
where
    T: Copy + Add + Mul + Zero + Debug + PartialEq + Send + Sync + Default,
    K: TilingKer<T>,
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

    unsafe fn a_from_packed(&self, ptr: *const T) -> StorageSpec<T> {
        StorageSpec::Packed { ptr, panel_len: (self.k * K::mr()) }
    }

    unsafe fn b_from_packed(&self, ptr: *const T) -> StorageSpec<T> {
        StorageSpec::Packed { ptr, panel_len: (self.k * K::nr()) }
    }

    unsafe fn b_from_data_and_offsets(
        &self,
        data: *const T,
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> StorageSpec<T> {
        debug_assert!(rows_offsets.len() > 0);
        debug_assert!(cols_offsets.len() > 0);
        let wanted = (cols_offsets.len() + K::nr() - 1) / K::nr() * K::nr();
        let mut col_ptrs: Vec<_> = Vec::with_capacity(wanted);
        col_ptrs.set_len(wanted);
        for i in 0..cols_offsets.len() {
            *col_ptrs.get_unchecked_mut(i) = data.offset(*cols_offsets.get_unchecked(i));
        }
        let pad = *col_ptrs.get_unchecked(cols_offsets.len() - 1);
        for i in cols_offsets.len()..wanted {
            *col_ptrs.get_unchecked_mut(i) = pad;
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
        StorageSpec::OffsetsAndPtrs { col_ptrs, row_byte_offsets, nr: K::nr() }
    }

    unsafe fn c_from_data_and_strides(
        &self,
        data: *const T,
        row_stride: isize,
        col_stride: isize,
    ) -> StorageSpec<T> {
        StorageSpec::Strides {
            ptr: data,
            row_byte_stride: row_stride * std::mem::size_of::<T>() as isize,
            col_byte_stride: col_stride * std::mem::size_of::<T>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn run(
        &self,
        a: &StorageSpec<T>,
        b: &StorageSpec<T>,
        c: &mut StorageSpec<T>,
        non_linear: &[NonLinearSpec<T>],
    ) {
        let mr = K::mr();
        let nr = K::nr();
        let m = self.m;
        let k = self.k;
        let n = self.n;
        let mut scratch = ScratchSpace::default();
        let tmpc = vec![T::zero(); mr * nr];
        let ref mut tmp_tile = self.c_from_data_and_strides(tmpc.as_ptr(), nr as isize, 1);
        let linear = LinearSpec::Mul { k };
        let linear = (&linear) as *const LinearSpec;
        for ia in 0..m / mr {
            let ref a = a.panel_a(ia);
            for ib in 0..n / nr {
                let ref b = b.panel_b(ib);
                let ref tile_c = c.tile(ia, ib);
                let non_linear = scratch.non_linear::<K>(non_linear, ia, ib);
                let err = K::kernel(&TileOpSpec {
                    a: a as _,
                    b: b as _,
                    c: tile_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
            }
            if n % nr != 0 {
                let ref b = b.panel_b(n / nr);
                let ref tmp_tile_c = tmp_tile.tile(0, 0);
                let non_linear = scratch.non_linear::<K>(non_linear, ia, n / nr);
                let err = K::kernel(&TileOpSpec {
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
            let ref tmp_tile_c = tmp_tile.tile(0, 0);
            for ib in 0..n / nr {
                let ref b = b.panel_b(ib);
                let non_linear = scratch.non_linear::<K>(non_linear, m / mr, ib);
                let err = K::kernel(&TileOpSpec {
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
                let ref b = b.panel_b(n / nr);
                let non_linear = scratch.non_linear::<K>(non_linear, m / mr, n / nr);
                let err = K::kernel(&TileOpSpec {
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

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::align;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;

    pub fn check_close(found: &[f32], expected: &[f32]) -> TestCaseResult {
        proptest::prop_assert!(
            found.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 0.001),
            "found: {:?} expected: {:?}",
            found,
            expected
        );
        Ok(())
    }

    #[macro_export]
    macro_rules! tile_frame_tests {
        ($cond:expr, $ker:ty) => {
            mod frame {
                #[allow(unused_imports)]
                use crate::frame::tiling::test::*;
                proptest::proptest! {
                    #[test]
                    fn mat_mul_prepacked((m, k, n, ref a, ref b) in strat_mat_mul()) {
                        if $cond {
                            test_mat_mul_prep_f32::<$ker>(m, k, n, a, b)?
                        }
                    }

                    #[test]
                    fn conv_prepacked(pb in strat_conv_1d()) {
                        if $cond {
                            check_close(&*pb.run::<$ker>(), &*pb.expected())?;
                        }
                    }
                }

                #[test]
                fn mat_mul_1() {
                    if $cond {
                        test_mat_mul_prep_f32::<$ker>(
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
                fn conv_prepacked_1() {
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
                        check_close(&*pb.run::<$ker>(), &*pb.expected()).unwrap();
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

    pub fn strat_mat_mul() -> BoxedStrategy<(usize, usize, usize, Vec<f32>, Vec<f32>)> {
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

    pub fn test_mat_mul_prep_f32<K: TilingKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        let op = TileOp::<K, f32>::new(m, k, n);
        unsafe {
            let mut packed_a: Vec<f32> =
                align::uninitialized(op.a_pack().len(), op.a_pack().alignment());
            op.a_pack().pack(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

            let mut packed_b: Vec<f32> =
                align::uninitialized(op.b_pack().len(), op.b_pack().alignment());
            op.b_pack().pack(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

            let mut found = vec![9999.0f32; m * n];

            op.run(
                &op.a_from_packed(packed_a.as_ptr()),
                &op.b_from_packed(packed_b.as_ptr()),
                &mut op.c_from_data_and_strides(found.as_mut_ptr(), n as isize, 1),
                &[],
            );

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

    pub unsafe fn fused_op<K: TilingKer<f32>, F: Fn(&mut [f32])>(
        m: usize,
        k: usize,
        n: usize,
        spec: &[NonLinearSpec<f32>],
        expect: F,
    ) -> proptest::test_runner::TestCaseResult {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];
        let op = TileOp::<K, f32>::new(m, k, n);

        let mut packed_a: Vec<f32> =
            align::uninitialized(op.a_pack().len(), op.a_pack().alignment());
        op.a_pack().pack(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

        let mut packed_b: Vec<f32> =
            align::uninitialized(op.b_pack().len(), op.b_pack().alignment());
        op.b_pack().pack(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

        let mut found = vec![9999.0f32; m * n];

        op.run(
            &op.a_from_packed(packed_a.as_ptr()),
            &op.b_from_packed(packed_b.as_ptr()),
            &mut op.c_from_data_and_strides(found.as_mut_ptr(), n as isize, 1),
            spec,
        );

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

    pub unsafe fn row_add<K: TilingKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        let bias = (0..m).map(|f| f as f32).collect::<Vec<f32>>();
        fused_op::<K, _>(m, k, n, &[NonLinearSpec::PerRowAdd(bias.clone())], |exp| {
            for x in 0..n {
                for y in 0..m {
                    exp[x + y * n] += bias[y]
                }
            }
        })
    }

    pub unsafe fn row_mul<K: TilingKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        let bias = (0..m).map(|f| f as f32).collect::<Vec<f32>>();
        fused_op::<K, _>(m, k, n, &[NonLinearSpec::PerRowMul(bias.clone())], |exp| {
            for x in 0..n {
                for y in 0..m {
                    exp[x + y * n] *= bias[y]
                }
            }
        })
    }

    pub unsafe fn col_add<K: TilingKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        let bias = (0..n).map(|f| f as f32).collect::<Vec<f32>>();
        fused_op::<K, _>(m, k, n, &[NonLinearSpec::PerColAdd(bias.clone())], |exp| {
            for x in 0..n {
                for y in 0..m {
                    exp[x + y * n] += bias[x]
                }
            }
        })
    }

    pub unsafe fn col_mul<K: TilingKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        let bias = (0..n).map(|f| f as f32).collect::<Vec<f32>>();
        fused_op::<K, _>(m, k, n, &[NonLinearSpec::PerColMul(bias.clone())], |exp| {
            for x in 0..n {
                for y in 0..m {
                    exp[x + y * n] *= bias[x]
                }
            }
        })
    }

    pub unsafe fn max<K: TilingKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        fused_op::<K, _>(m, k, n, &[NonLinearSpec::Max(5f32)], |exp| {
            exp.iter_mut().for_each(|x| *x = x.max(5f32))
        })
    }

    pub unsafe fn min<K: TilingKer<f32>>(
        m: usize,
        k: usize,
        n: usize,
    ) -> proptest::test_runner::TestCaseResult {
        fused_op::<K, _>(m, k, n, &[NonLinearSpec::Min(1f32)], |exp| {
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

        pub fn run<K: TilingKer<f32>>(&self) -> Vec<f32> {
            let op = TileOp::<K, f32>::new(self.m(), self.k(), self.n());
            unsafe {
                let mut packed_a: Vec<f32> =
                    align::uninitialized(op.a_pack().len(), op.a_pack().alignment());
                op.a_pack().pack(
                    packed_a.as_mut_ptr(),
                    self.filters.as_ptr(),
                    self.k() as isize,
                    1,
                );

                let mut found = vec![9999.0f32; self.co * self.output_width()];
                op.run(
                    &op.a_from_packed(packed_a.as_ptr()),
                    &op.b_from_data_and_offsets(
                        self.data.as_ptr(),
                        &self.data_rows_offsets(),
                        &self.data_cols_offsets(),
                    ),
                    &mut op.c_from_data_and_strides(found.as_mut_ptr(), self.n() as isize, 1),
                    &[],
                );
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
