use std::borrow::Cow;
use std::fmt::Debug;

use crate::frame::mmm::FusedKerSpec;
use crate::LADatum;

use super::FusedSpec;

pub trait MatMatMulKer: Copy + Clone + Debug + Send + Sync + 'static {
    type Acc: LADatum;
    fn name(&self) -> Cow<'static, str>;
    fn kernel(&self, op: &[FusedKerSpec<Self::Acc>]) -> isize;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;
    fn alignment_bytes_packed_a(&self) -> usize;
    fn end_padding_packed_a(&self) -> usize;
    fn alignment_bytes_packed_b(&self) -> usize;
    fn end_padding_packed_b(&self) -> usize;

    #[allow(unused_variables)]
    fn prefetch(&self, ptr: *const u8, len: usize) {}

    #[allow(unused_variables)]
    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        true
    }
}

#[macro_export]
macro_rules! test_mmm_kernel_f16 {
    ($k: ident, $cond: expr) => {
        paste! {
            #[cfg(test)]
            #[allow(non_snake_case)]
            mod [<test_ $k>] {
                use super::$k;
                mmm_kernel_tests!($cond, $k, f16, f16, f16, f16);
                mmm_frame_tests!($cond, $k, f16, f16, f16, f16);
                mmm_kernel_fuse_tests!($cond, $k, f16, f16);
            }
        }
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f32 {
    ($k: ident, $cond: expr) => {
        paste! {
            #[cfg(test)]
            #[allow(non_snake_case)]
            mod [<test_ $k>] {
                use super::$k;
                mmm_kernel_tests!($cond, $k, f32, f32, f32, f32);
                mmm_frame_tests!($cond, $k, f32, f32, f32, f32);
                mmm_kernel_fuse_tests!($cond, $k, f32, f32);
            }
        }
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f64 {
    ($k: ident, $cond: expr) => {
        paste! {
            #[cfg(test)]
            #[allow(non_snake_case)]
            mod [<test_ $k>] {
                use super::$k;
                mmm_kernel_tests!($cond, $k, f64, f64, f64, f64);
                mmm_frame_tests!($cond, $k, f64, f64, f64, f64);
                mmm_kernel_fuse_tests!($cond, $k, f64, f64);
            }
        }
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_i32 {
    ($k: ident, $cond: expr) => {
        paste! {
            #[cfg(test)]
            #[allow(non_snake_case)]
            mod [<test_ $k>] {
                use super::$k;
                mmm_kernel_tests!($cond, $k, i8, i8, i8, i32);
                mmm_kernel_fuse_tests!($cond, $k, i8, i32);
                mmm_frame_tests!($cond, $k, i8, i8, i8, i32);
            }
            #[cfg(test)]
            mod [<test_qi8_ $k>] {
                use super::$k;
                qmmm_kernel_fuse_tests!($cond, $k, i8, i8, i8, i32);
            }
            #[cfg(test)]
            mod [<test_qi32_ $k>] {
                use super::$k;
                qmmm_kernel_fuse_tests!($cond, $k, i8, i8, i32, i32);
            }
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::frame::mmm::OutputStoreKer;
    use num_traits::{AsPrimitive, One, Zero};
    use proptest::collection::vec;
    use proptest::prelude::*;
    use std::fmt;
    use std::marker::PhantomData;
    use tract_data::internal::*;

    #[macro_export]
    macro_rules! mmm_kernel_tests {
        ($cond:expr, $ker:ident, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
            mod kernel {
                use super::super::$ker;
                use num_traits::Zero;
                use proptest::prelude::*;
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                #[allow(unused_imports)]
                use $crate::frame::mmm::kernel::test;
                use $crate::frame::mmm::kernel::test::PackedPackedProblem;
                use $crate::frame::mmm::kernel::MatMatMulKer;

                proptest::proptest! {
                    #[test]
                    fn packed_packed_prop(pb in any_with::<PackedPackedProblem<_, $ta, $tb, $tc, $ti>>($ker)) {
                        if $cond {
                            prop_assert_eq!(pb.run(), pb.reference())
                        }
                    }
                }

                #[test]
                fn packed_packed_1() {
                    if $cond {
                        test::packed_packed::<_, $ta, $tb, $tc, $ti>($ker, 1)
                    }
                }

                #[test]
                fn packed_packed_2() {
                    if $cond {
                        test::packed_packed::<_, $ta, $tb, $tc, $ti>($ker, 2)
                    }
                }

                #[test]
                fn packed_packed_13() {
                    if $cond {
                        test::packed_packed::<_, $ta, $tb, $tc, $ti>($ker, 13)
                    }
                }

                #[test]
                fn packed_packed_empty() {
                    if $cond {
                        let pb = PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                            $ker,
                            0,
                            vec![<$ta>::zero(); 0],
                            vec![<$tb>::zero(); 0],
                            false,
                            false,
                        );
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn packed_packed_bug_1() {
                    if $cond {
                        let pb = PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                            $ker,
                            1,
                            vec![<$ta>::zero(); $ker.mr()],
                            vec![<$tb>::zero(); $ker.nr()],
                            true,
                            true,
                        );
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn packed_vec_k1() {
                    if $cond {
                        test::packed_vec::<_, $ta, $tb, $tc, $ti>($ker, 1)
                    }
                }

                #[test]
                fn packed_vec_k2() {
                    if $cond {
                        test::packed_vec::<_, $ta, $tb, $tc, $ti>($ker, 2)
                    }
                }

                #[test]
                fn packed_vec_k4() {
                    if $cond {
                        test::packed_vec::<_, $ta, $tb, $tc, $ti>($ker, 4)
                    }
                }

                #[test]
                fn packed_vec_k13() {
                    if $cond {
                        test::packed_vec::<_, $ta, $tb, $tc, $ti>($ker, 13)
                    }
                }
            }
        };
    }

    #[derive(Debug, new)]
    pub struct PackedPackedProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<Acc = TI>,
        TA: 'static + Debug + AsPrimitive<TI>,
        TB: 'static + Debug + AsPrimitive<TI>,
        TC: Copy + PartialEq + 'static + Debug,
        TI: LADatum + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        pub ker: K,
        pub k: usize,
        pub a: Vec<TA>,
        pub b: Vec<TB>,
        pub trans_c: bool,
        pub add_one: bool,
        pub _phantom: PhantomData<(K, TC, TI)>,
    }

    impl<K, TA, TB, TC, TI> Arbitrary for PackedPackedProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<Acc = TI> + Default,
        TA: 'static + Debug + AsPrimitive<TI>,
        TB: 'static + Debug + AsPrimitive<TI>,
        TC: Copy + PartialEq + 'static + Debug,
        TI: LADatum + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        type Parameters = K;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: K) -> Self::Strategy {
            (0usize..20, any::<bool>(), any::<bool>())
                .prop_flat_map(|(k, trans_c, add_one)| {
                    let ker = K::default();
                    let m = k * ker.mr();
                    let n = k * ker.nr();
                    let a = (0usize..10).prop_map(|x| x.as_());
                    let b = (0usize..10).prop_map(|x| x.as_());
                    (Just(k), Just(trans_c), Just(add_one), vec(a, m..=m), vec(b, n..=n))
                })
                .prop_map(|(k, trans_c, add_one, a, b)| Self {
                    ker: K::default(),
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
        K: MatMatMulKer<Acc = TI>,
        TA: 'static + Debug + AsPrimitive<TI> + Datum,
        TB: 'static + Debug + AsPrimitive<TI> + Datum,
        TC: Copy + Zero + PartialEq + 'static + Debug,
        TI: LADatum + fmt::Display + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        pub fn reference(&self) -> Vec<TC> {
            let init = if self.add_one { TI::one() } else { TI::zero() };
            let mr = self.ker.mr();
            let nr = self.ker.nr();
            let mut vi = vec![init; mr * nr];
            for m in 0..mr {
                for n in 0..nr {
                    for k in 0..self.k {
                        let a: TI = self.a[m + mr * k].as_();
                        let b: TI = self.b[n + nr * k].as_();
                        let offset = if self.trans_c { m + n * mr } else { n + m * nr };
                        vi[offset] += a * b;
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
                    .chain(vec![0.as_(); self.ker.end_padding_packed_a() * self.ker.mr()])
                    .collect::<Vec<_>>();
                let pa = Tensor::from_slice_align(&a, self.ker.alignment_bytes_packed_a()).unwrap();
                let b = self
                    .b
                    .iter()
                    .cloned()
                    .chain(vec![0.as_(); self.ker.end_padding_packed_b() * self.ker.nr()])
                    .collect::<Vec<_>>();
                let pb = Tensor::from_slice_align(&b, self.ker.alignment_bytes_packed_b()).unwrap();
                let mut v = vec![TC::zero(); self.ker.mr() * self.ker.nr()];
                let c = if self.trans_c {
                    mmm_stride_storage(&mut v, 1, self.ker.mr())
                } else {
                    mmm_stride_storage(&mut v, self.ker.nr(), 1)
                };
                let b_store = pb.as_ptr_unchecked::<TB>() as _;

                let mut non_linear_ops = tvec!(FusedKerSpec::AddMatMul {
                    k: self.k,
                    pa: pa.as_ptr_unchecked::<u8>() as _,
                    pb: b_store,
                    packing: 0,
                });
                if self.add_one {
                    non_linear_ops.push(FusedKerSpec::ScalarAdd(TI::one()));
                }
                non_linear_ops.push(FusedKerSpec::Store(c));
                non_linear_ops.push(FusedKerSpec::Done);
                non_linear_ops.insert(0, FusedKerSpec::Clear);
                let err = self.ker.kernel(&non_linear_ops);
                assert_eq!(err, 0);
                v
            }
        }
    }

    pub fn packed_packed<K, TA, TB, TC, TI>(ker: K, k: usize)
    where
        K: MatMatMulKer<Acc = TI>,
        TA: Copy + One + Datum + AsPrimitive<TI>,
        TB: Copy + One + Datum + AsPrimitive<TI>,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TA> + AsPrimitive<TB>,
    {
        let a = vec![TA::one(); ker.mr() * k];
        let b = vec![TB::one(); ker.nr() * k];
        let pb = PackedPackedProblem::<K, TA, TB, TC, TI>::new(ker, k, a, b, false, false);
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

    pub fn packed_vec<K, TA, TB, TC, TI>(ker: K, k: usize)
    where
        K: MatMatMulKer<Acc = TI>,
        TA: Copy + One + AsPrimitive<TI> + Debug + Datum,
        TB: Copy + One + AsPrimitive<TI> + Debug + Datum,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC>,
    {
        let pa = unsafe {
            Tensor::from_slice_align(
                &vec![TA::one(); ker.mr() * (k + ker.end_padding_packed_a())],
                ker.alignment_bytes_packed_a(),
            )
            .unwrap()
        };
        let b = vec![TB::one(); (k + 1) * ker.nr()];
        let mut c: Vec<TC> = vec![TC::zero(); ker.mr() * ker.nr()];
        let tile = mmm_stride_storage(&mut c, 1, 0);
        let pb = unsafe { Tensor::from_slice_align(&b, ker.alignment_bytes_packed_b()).unwrap() };
        let non_linear_ops = tvec!(
            FusedKerSpec::Clear,
            FusedKerSpec::AddMatMul {
                pa: unsafe { pa.as_ptr_unchecked::<u8>() as _ },
                pb: unsafe { pb.as_ptr_unchecked::<u8>() as _ },
                k,
                packing: 0,
            },
            FusedKerSpec::Store(tile),
            FusedKerSpec::Done
        );
        let err = ker.kernel(&non_linear_ops);
        assert_eq!(err, 0);
        let expected = vec![k.as_(); ker.mr()];
        assert_eq!(c[..ker.mr()], expected);
    }
}
