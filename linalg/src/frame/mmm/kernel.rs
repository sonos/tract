use std::fmt::Debug;

use crate::frame::mmm::FusedKerSpec;

pub trait MatMatMulKer<TI>: Copy + Clone + Debug + Send + Sync + 'static
where
    TI: Copy + Debug,
{
    fn name() -> &'static str;
    fn kernel(op: &[FusedKerSpec<TI>]) -> isize;
    fn mr() -> usize;
    fn nr() -> usize;
    fn alignment_bytes_packed_a() -> usize;
    fn end_padding_packed_a() -> usize;
    fn alignment_bytes_packed_b() -> usize;
    fn end_padding_packed_b() -> usize;

    #[allow(unused_variables)]
    fn prefetch(ptr: *const u8, len: usize) {
    }
}

#[macro_export]
macro_rules! test_mmm_kernel_f32 {
    ($k: ty, $id: ident, $cond: expr) => {
        #[cfg(test)]
        #[allow(non_snake_case)]
        mod $id {
            mmm_kernel_tests!($cond, $k, f32, f32, f32, f32);
            mmm_frame_tests!($cond, $k, f32, f32, f32, f32);
            mmm_kernel_fuse_tests!($cond, $k, f32, f32);
        }
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_i8 {
    ($k: ty, $id: ident, $cond: expr) => {
        #[cfg(test)]
        #[allow(non_snake_case)]
        mod $id {
            mmm_kernel_tests!($cond, $k, i8, i8, i8, i32);
            mmm_kernel_fuse_tests!($cond, $k, i8, i32);
            mmm_frame_tests!($cond, $k, i8, i8, i8, i32);
            qmmm_kernel_fuse_tests!($cond, $k, i8, i8, i8, i32);
        }
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_i8_i32 {
    ($k: ty, $id: ident, $cond: expr) => {
        #[cfg(test)]
        #[allow(non_snake_case)]
        mod $id {
            mmm_kernel_tests!($cond, $k, i8, i8, i32, i32);
            mmm_kernel_fuse_tests!($cond, $k, i32, i32);
            mmm_frame_tests!($cond, $k, i8, i8, i32, i32);
            qmmm_kernel_fuse_tests!($cond, $k, i8, i8, i32, i32);
        }
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_i8_u8_i32 {
    ($k: ty, $id: ident, $cond: expr) => {
        #[cfg(test)]
        #[allow(non_snake_case)]
        mod $id {
            mmm_kernel_tests!($cond, $k, i8, u8, i32, i32);
            mmm_kernel_fuse_tests!($cond, $k, i32, i32);
            mmm_frame_tests!($cond, $k, i8, u8, i32, i32);
            qmmm_kernel_fuse_tests!($cond, $k, i8, u8, i32, i32);
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::frame::mmm::{ InputStoreKer, PackedStoreKer, OutputStoreKer };
    use num_traits::{AsPrimitive, One, Zero};
    use proptest::collection::vec;
    use proptest::prelude::*;
    use std::fmt;
    use std::marker::PhantomData;
    use std::ops::{Add, Mul};
    use tract_data::internal::*;

    #[macro_export]
    macro_rules! mmm_kernel_tests {
        ($cond:expr, $ker:ty, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
            mod kernel {
                use num_traits::Zero;
                use proptest::prelude::*;
                #[allow(unused_imports)]
                use crate::frame::mmm::kernel::test;
                use crate::frame::mmm::kernel::test::{ PackedOffsetsProblem, PackedPackedProblem };
                use crate::frame::mmm::MatMatMulKer;

                proptest::proptest! {
                    #[test]
                    fn packed_packed_prop(pb in any::<PackedPackedProblem<$ker, $ta, $tb, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.run(), pb.reference())
                        }
                    }

                    #[test]
                    fn packed_offsets_prop(pb in any::<PackedOffsetsProblem<$ker, $ta, $tb, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.run(), pb.reference())
                        }
                    }
                }

                #[test]
                fn packed_packed_1() {
                    if $cond {
                        test::packed_packed::<$ker, $ta, $tb, $tc, $ti>(1)
                    }
                }

                #[test]
                fn packed_packed_2() {
                    if $cond {
                        test::packed_packed::<$ker, $ta, $tb, $tc, $ti>(2)
                    }
                }

                #[test]
                fn packed_packed_13() {
                    if $cond {
                        test::packed_packed::<$ker, $ta, $tb, $tc, $ti>(13)
                    }
                }

                #[test]
                fn packed_packed_bug_1() {
                    if $cond {
                        let pb = PackedPackedProblem::<$ker, $ta, $tb, $tc, $ti>::new(
                            1,
                            vec!(<$ta>::zero(); <$ker>::mr()),
                            vec!(<$tb>::zero(); <$ker>::nr()),
                            true,
                            true);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn packed_offsets_bug_1() {
                    if $cond {
                        let pb = PackedOffsetsProblem::<$ker, $ta, $tb, $tc, $ti>::new(
                            vec!(<$ta>::zero(); <$ker>::mr()),
                            vec!(<$tb>::zero()),
                            vec!(0usize; <$ker>::nr()),
                            vec!(0usize),
                            true);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn packed_offsets_bug_2() {
                    if $cond {
                        let mut pa = vec!(<$ta>::zero(); <$ker>::mr() * 3);
                        let len = pa.len() - 1;
                        pa[len] = 1 as _;
                        let pb = PackedOffsetsProblem::<$ker, $ta, $tb, $tc, $ti>::new(pa,
                                                                                       vec!(0 as _, 1 as _),
                                                                                       vec!(0usize; <$ker>::nr()),
                                                                                       vec!(1usize, 0, 0),
                                                                                       true);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }


                #[test]
                fn packed_offsets_k1() {
                    if $cond {
                        test::packed_offsets::<$ker, $ta, $tb, $tc, $ti>(1, <$ker>::nr())
                    }
                }

                #[test]
                fn packed_offsets_k2() {
                    if $cond {
                        test::packed_offsets::<$ker, $ta, $tb, $tc, $ti>(2, <$ker>::nr())
                    }
                }

                #[test]
                fn packed_offsets_k13() {
                    if $cond {
                        test::packed_offsets::<$ker, $ta, $tb, $tc, $ti>(13, <$ker>::nr())
                    }
                }

                #[test]
                fn packed_vec_k1() {
                    if $cond {
                        test::packed_vec::<$ker, $ta, $tb, $tc, $ti>(1)
                    }
                }

                #[test]
                fn packed_vec_k2() {
                    if $cond {
                        test::packed_vec::<$ker, $ta, $tb, $tc, $ti>(2)
                    }
                }

                #[test]
                fn packed_vec_k4() {
                    if $cond {
                        test::packed_vec::<$ker, $ta, $tb, $tc, $ti>(4)
                    }
                }

                #[test]
                fn packed_vec_k13() {
                    if $cond {
                        test::packed_vec::<$ker, $ta, $tb, $tc, $ti>(13)
                    }
                }

                #[test]
                fn packed_offsets_with_row_stride() {
                    if $cond {
                        test::packed_offsets::<$ker, $ta, $tb, $tc, $ti>(2, <$ker>::nr() + 5)
                    }
                }
            }
        };
    }

    #[derive(Debug, new)]
    pub struct PackedPackedProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: 'static + Debug + AsPrimitive<TI>,
        TB: 'static + Debug + AsPrimitive<TI>,
        TC: Copy + PartialEq + 'static + Debug,
        TI: Copy + Add + Mul + Zero + Debug + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        pub k: usize,
        pub a: Vec<TA>,
        pub b: Vec<TB>,
        pub trans_c: bool,
        pub add_one: bool,
        pub _phantom: PhantomData<(K, TC, TI)>,
    }

    impl<K, TA, TB, TC, TI> Arbitrary for PackedPackedProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: 'static + Debug + AsPrimitive<TI>,
        TB: 'static + Debug + AsPrimitive<TI>,
        TC: Copy + PartialEq + 'static + Debug,
        TI: Copy + Add + Mul + Zero + Debug + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (0usize..20, any::<bool>(), any::<bool>())
                .prop_flat_map(|(k, trans_c, add_one)| {
                    let m = k * K::mr();
                    let n = k * K::nr();
                    let a = (0usize..10).prop_map(|x| x.as_());
                    let b = (0usize..10).prop_map(|x| x.as_());
                    (Just(k), Just(trans_c), Just(add_one), vec(a, m..=m), vec(b, n..=n))
                })
                .prop_map(|(k, trans_c, add_one, a, b)| Self {
                    k,
                    a,
                    b,
                    trans_c,
                    add_one,
                    _phantom: PhantomData,
                })
                .boxed()
        }
    }

    impl<K, TA, TB, TC, TI> PackedPackedProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: 'static + Debug + AsPrimitive<TI> + Datum,
        TB: 'static + Debug + AsPrimitive<TI> + Datum,
        TC: Copy + Zero + PartialEq + 'static + Debug,
        TI: Copy + Add + Mul<Output = TI> + Zero + One + Debug + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        pub fn reference(&self) -> Vec<TC> {
            let init = if self.add_one { TI::one() } else { TI::zero() };
            let mut vi = vec![init; K::mr() * K::nr()];
            let mr = K::mr();
            let nr = K::nr();
            for m in 0..mr {
                for n in 0..nr {
                    for k in 0..self.k {
                        let a: TI = self.a[m + mr * k].as_();
                        let b: TI = self.b[n + nr * k].as_();
                        let offset = if self.trans_c { m + n * mr } else { n + m * nr };
                        vi[offset] = vi[offset] + a * b;
                    }
                }
            }
            vi.into_iter().map(|ti| ti.as_()).collect()
        }

        pub fn run(&self) -> Vec<TC> {
            unsafe {
                let a = self
                    .a
                    .iter()
                    .cloned()
                    .chain(vec![0.as_(); K::end_padding_packed_a() * K::mr()])
                    .collect::<Vec<_>>();
                let pa = Tensor::from_slice_align(&a, K::alignment_bytes_packed_a()).unwrap();
                let b = self
                    .b
                    .iter()
                    .cloned()
                    .chain(vec![0.as_(); K::end_padding_packed_b() * K::nr()])
                    .collect::<Vec<_>>();
                let pb = Tensor::from_slice_align(&b, K::alignment_bytes_packed_b()).unwrap();
                let mut v = vec![TC::zero(); K::mr() * K::nr()];
                let c = if self.trans_c {
                    mmm_stride_storage(&mut v, 1, K::mr())
                } else {
                    mmm_stride_storage(&mut v, K::nr(), 1)
                };
                let b_store =
                    InputStoreKer::Packed(PackedStoreKer { ptr: pb.as_ptr_unchecked::<TB>() as _ });

                let mut non_linear_ops = tvec!(FusedKerSpec::AddMatMul {
                    k: self.k,
                    pa: pa.as_ptr_unchecked::<u8>() as _,
                    pb: &b_store,
                    cpu_variant: 0,
                });
                if self.add_one {
                    non_linear_ops.push(FusedKerSpec::ScalarAdd(TI::one()));
                }
                non_linear_ops.push(FusedKerSpec::Store(c));
                non_linear_ops.push(FusedKerSpec::Done);
                let err = K::kernel(&non_linear_ops);
                assert_eq!(err, 0);
                v
            }
        }
    }

    #[derive(Debug, new)]
    pub struct PackedOffsetsProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: 'static + Debug + AsPrimitive<TI>,
        TB: 'static + Debug + AsPrimitive<TI>,
        TC: Copy + PartialEq + 'static + Debug,
        TI: Copy + Add + Mul + Zero + Debug + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        a: Vec<TA>,
        b: Vec<TB>,
        cols_offsets: Vec<usize>,
        rows_offsets: Vec<usize>,
        add_one: bool,
        _phantom: PhantomData<(K, TC, TI)>,
    }

    impl<K, TA, TB, TC, TI> Arbitrary for PackedOffsetsProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: 'static + Debug + AsPrimitive<TI>,
        TB: 'static + Debug + AsPrimitive<TI>,
        TC: Copy + PartialEq + 'static + Debug,
        TI: Copy + Add + Mul + Zero + Debug + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (vec(0usize..100, 1usize..20), vec(0usize..100, K::nr()..=K::nr()), any::<bool>())
                .prop_flat_map(|(rows_offsets, cols_offsets, add_one)| {
                    let k = rows_offsets.len();
                    let m = k * K::mr();
                    let a = (0usize..10).prop_map(|x| x.as_());
                    let b = (0usize..10).prop_map(|x| x.as_());
                    let len =
                        rows_offsets.iter().max().unwrap() + cols_offsets.iter().max().unwrap() + 1;
                    (
                        vec(a, m..=m),
                        vec(b, len..=len),
                        Just(rows_offsets),
                        Just(cols_offsets),
                        Just(add_one),
                    )
                })
                .prop_map(|(a, b, rows_offsets, cols_offsets, add_one)| Self {
                    a,
                    b,
                    rows_offsets,
                    cols_offsets,
                    add_one,
                    _phantom: PhantomData,
                })
                .boxed()
        }
    }

    impl<K, TA, TB, TC, TI> PackedOffsetsProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: 'static + Debug + AsPrimitive<TI> + Datum,
        TB: 'static + Debug + AsPrimitive<TI> + Datum,
        TC: Copy + Zero + PartialEq + 'static + Debug,
        TI: Copy + Add + Mul<Output = TI> + Zero + One + Debug + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        pub fn reference(&self) -> Vec<TC> {
            let init = if self.add_one { TI::one() } else { TI::zero() };
            let mut vi = vec![init; K::mr() * K::nr()];
            let mr = K::mr();
            let nr = K::nr();
            for m in 0..mr {
                for n in 0..nr {
                    for k in 0..self.rows_offsets.len() {
                        let a: TI = self.a[m + mr * k].as_();
                        let b: TI = self.b[self.rows_offsets[k] + self.cols_offsets[n]].as_();
                        let offset = n + m * nr;
                        vi[offset] = vi[offset] + a * b;
                    }
                }
            }
            vi.into_iter().map(|ti| ti.as_()).collect()
        }

        pub fn run(&self) -> Vec<TC> {
            let a = self
                .a
                .iter()
                .cloned()
                .chain(vec![0.as_(); K::end_padding_packed_a() * K::mr()])
                .collect::<Vec<_>>();
            let pa =
                unsafe { Tensor::from_slice_align(&a, K::alignment_bytes_packed_a()).unwrap() };
            let rows_offset: Vec<isize> = self
                .rows_offsets
                .iter()
                .map(|o| (o * std::mem::size_of::<TB>()) as isize)
                .collect();
            let col_ptrs: Vec<*const TB> = unsafe {
                self.cols_offsets.iter().map(|o| self.b.as_ptr().offset(*o as isize)).collect()
            };
            let mut v = vec![TC::zero(); K::mr() * K::nr()];
            let c = mmm_stride_storage(&mut v, K::nr(), 1);
            let b_store = InputStoreKer::OffsetsAndPtrs {
                row_byte_offsets: rows_offset.as_ptr(),
                col_ptrs: col_ptrs.as_ptr() as _,
            };

            let mut non_linear_ops = tvec!(FusedKerSpec::AddMatMul {
                k: self.rows_offsets.len(),
                pa: unsafe { pa.as_ptr_unchecked::<u8>() as _ },
                pb: &b_store,
                cpu_variant: 0,
            });
            if self.add_one {
                non_linear_ops.push(FusedKerSpec::ScalarAdd(TI::one()));
            }
            non_linear_ops.push(FusedKerSpec::Store(c));
            non_linear_ops.push(FusedKerSpec::Done);
            let err = K::kernel(&non_linear_ops);
            assert_eq!(err, 0);
            v
        }
    }

    pub fn packed_packed<K, TA, TB, TC, TI>(k: usize)
    where
        K: MatMatMulKer<TI>,
        TA: Copy + One + Datum + AsPrimitive<TI>,
        TB: Copy + One + Datum + AsPrimitive<TI>,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy
            + Add
            + Mul<Output = TI>
            + Zero
            + One
            + Debug
            + fmt::Display
            + 'static
            + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TA> + AsPrimitive<TB>,
    {
        let a = vec![TA::one(); K::mr() * k];
        let b = vec![TB::one(); K::nr() * k];
        let pb = PackedPackedProblem::<K, TA, TB, TC, TI>::new(k, a, b, false, false);
        assert_eq!(pb.run(), pb.reference())
    }

    pub fn mmm_stride_storage<T: Copy>(v: &mut [T], rsc: usize, csc: usize) -> OutputStoreKer {
        OutputStoreKer {
            ptr: v.as_mut_ptr() as _,
            row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
            col_byte_stride: (std::mem::size_of::<T>() * csc) as isize,
            item_size: std::mem::size_of::<T>(),
        }
    }

    pub fn packed_offsets<K, TA, TB, TC, TI>(k: usize, t: usize)
    where
        K: MatMatMulKer<TI>,
        TA: Copy + One + AsPrimitive<TI> + Datum,
        TB: Copy + One + AsPrimitive<TI> + Datum,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy + Add + Zero + Mul<Output = TI> + Debug + fmt::Display + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        let a: Vec<TA> = (1..=(k * K::mr())).map(|x| x.as_()).collect();
        let pa = unsafe { Tensor::from_slice_align(&a, K::alignment_bytes_packed_a()).unwrap() };
        let b: Vec<TB> = (0..(k * t)).map(|x| x.as_()).collect();
        let len = K::mr() * K::nr();
        let mut v: Vec<TC> = vec![TC::zero(); len];
        let c = mmm_stride_storage(&mut v, K::nr(), 1);
        let col_ptrs = (0..K::nr()).map(|i| (&b[i]) as *const TB as _).collect::<Vec<_>>();
        let row_byte_offsets =
            (0..k).map(|i| (i * std::mem::size_of::<TB>() * t) as isize).collect::<Vec<_>>();
        let b_store = InputStoreKer::OffsetsAndPtrs {
            row_byte_offsets: row_byte_offsets.as_ptr(),
            col_ptrs: col_ptrs.as_ptr(),
        };

        let mut non_linear_ops = tvec!(FusedKerSpec::AddMatMul {
            k,
            pa: unsafe { pa.as_ptr_unchecked::<u8>() as _ },
            pb: &b_store,
            cpu_variant: 0,
        });
        non_linear_ops.push(FusedKerSpec::Store(c));
        non_linear_ops.push(FusedKerSpec::Done);
        let err = K::kernel(&non_linear_ops);
        assert_eq!(err, 0);
        let expected: Vec<TC> = (0..v.len())
            .map(|ix| {
                let row = ix / K::nr();
                let col = ix % K::nr();
                (0..k)
                    .map(|i| {
                        pa.as_slice::<TA>().unwrap()[K::mr() * i + row].as_() * b[t * i + col].as_()
                    })
                    .fold(TI::zero(), |s, a| s + a)
                    .as_()
            })
            .collect();
        assert_eq!(v, expected);
    }

    pub fn packed_vec<K, TA, TB, TC, TI>(k: usize)
    where
        K: MatMatMulKer<TI>,
        TA: Copy + One + AsPrimitive<TI> + Debug + Datum,
        TB: Copy + One + AsPrimitive<TI> + Debug + Datum,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy + Add + Zero + Mul<Output = TI> + Debug + fmt::Display + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TC>,
    {
        let pa = unsafe {
            Tensor::from_slice_align(
                &vec![TA::one(); K::mr() * (k + K::end_padding_packed_a())],
                K::alignment_bytes_packed_a(),
            )
            .unwrap()
        };
        let b = vec![TB::one(); (k + 1) * K::nr()];
        let mut c: Vec<TC> = vec![TC::zero(); K::mr() * K::nr()];
        let mut non_linear_ops = tvec!();
        let tile = mmm_stride_storage(&mut c, 1, 0);
        let b_store = InputStoreKer::Packed(PackedStoreKer { ptr: b.as_ptr() as _ });
        non_linear_ops.push(FusedKerSpec::AddMatMul {
            pa: unsafe { pa.as_ptr_unchecked::<u8>() as _ },
            pb: &b_store,
            k,
            cpu_variant: 0,
        });
        non_linear_ops.push(FusedKerSpec::Store(tile));
        non_linear_ops.push(FusedKerSpec::Done);
        let err = K::kernel(&non_linear_ops);
        assert_eq!(err, 0);
        let expected = vec![k.as_(); K::mr()];
        assert_eq!(c[..K::mr()], expected);
    }
}
