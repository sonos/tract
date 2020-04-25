use std::fmt::Debug;

use super::*;

#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct MatMatMulKerSpec<'a, TA, TB, TC, TI>
where
    TA: Copy,
    TB: Copy,
    TC: Copy,
    TI: Copy + Debug,
{
    pub a: &'a PanelStore<TA>,
    pub b: &'a PanelStore<TB>,
    pub c: &'a PanelStore<TC>,
    pub linear: &'a LinearSpec,
    pub non_linear: *const FusedKerSpec<TI>,
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum LinearSpec {
    Mul { k: usize },
    Noop,
}

impl LinearSpec {
    pub fn k(k: usize) -> LinearSpec {
        LinearSpec::Mul { k }
    }
}

pub trait MatMatMulKer<TA, TB, TC, TI>: Copy + Clone + Debug + Send + Sync + 'static
where
    TA: Copy,
    TB: Copy,
    TC: Copy,
    TI: Copy + Debug,
{
    fn name() -> &'static str;
    fn kernel(op: &MatMatMulKerSpec<TA, TB, TC, TI>) -> isize;
    fn mr() -> usize;
    fn nr() -> usize;
    fn alignment_bytes_packed_a() -> usize;
    fn alignment_bytes_packed_b() -> usize;
}

#[macro_export]
macro_rules! test_mmm_kernel_f32 {
    ($k: ty, $id: ident, $cond: expr) => {
        #[cfg(test)]
        #[allow(non_snake_case)]
        mod $id {
            mmm_kernel_tests!($cond, $k, f32, f32, f32, f32);
            mmm_frame_tests!($cond, $k, f32, f32, f32, f32);
            mmm_kernel_fuse_tests!($cond, $k, f32, f32, f32, f32);
            mmm_s_frame_tests!($cond, $k, f32, f32, f32, f32);
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
            mmm_kernel_fuse_tests!($cond, $k, i8, i8, i8, i32);
            mmm_frame_tests!($cond, $k, i8, i8, i8, i32);
            mmm_s_frame_tests!($cond, $k, i8, i8, i8, i32);
            qmmm_kernel_fuse_tests!($cond, $k, i8, i8, i8, i32);
            qmmm_frame_tests!($cond, $k, i8, i8, i8, i32);
            qmmm_s_frame_tests!($cond, $k, i8, i8, i8, i32);
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
            mmm_kernel_fuse_tests!($cond, $k, i8, i8, i32, i32);
            mmm_frame_tests!($cond, $k, i8, i8, i32, i32);
            mmm_s_frame_tests!($cond, $k, i8, i8, i32, i32);
            qmmm_kernel_fuse_tests!($cond, $k, i8, i8, i32, i32);
            qmmm_frame_tests!($cond, $k, i8, i8, i32, i32);
        }
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_u8 {
    ($k: ty, $id: ident, $cond: expr) => {
        #[cfg(test)]
        #[allow(non_snake_case)]
        mod $id {
            mmm_kernel_tests!($cond, $k, u8, u8, u8, i32);
            mmm_kernel_fuse_tests!($cond, $k, u8, u8, u8, i32);
            qmmm_kernel_fuse_tests!($cond, $k, u8, u8, u8, i32);
            qmmm_frame_tests!($cond, $k, u8, u8, u8, i32);
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::align::Buffer;
    use num_traits::{AsPrimitive, One, Zero};
    use proptest::collection::vec;
    use proptest::prelude::*;
    use std::fmt;
    use std::marker::PhantomData;
    use std::ops::{Add, Mul};

    #[test]
    fn check_non_linear_enum_size() {
        assert_eq!(
            std::mem::size_of::<super::FusedKerSpec<f32>>(),
            3 * std::mem::size_of::<usize>()
        )
    }

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
        K: MatMatMulKer<TA, TB, TC, TI>,
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
        K: MatMatMulKer<TA, TB, TC, TI>,
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
        K: MatMatMulKer<TA, TB, TC, TI>,
        TA: 'static + Debug + AsPrimitive<TI>,
        TB: 'static + Debug + AsPrimitive<TI>,
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
            let pa = Buffer::realign_data(&self.a, K::alignment_bytes_packed_a());
            let pb = Buffer::realign_data(&self.b, K::alignment_bytes_packed_b());
            let mut v = vec![TC::zero(); K::mr() * K::nr()];
            let mut c = if self.trans_c {
                mmm_stride_storage(&mut v, 1, K::mr())
            } else {
                mmm_stride_storage(&mut v, K::nr(), 1)
            };
            let non_linear_ops = [FusedKerSpec::ScalarAdd(TI::one()), FusedKerSpec::Done];
            let non_linear = if self.add_one {
                non_linear_ops.as_ptr()
            } else {
                std::ptr::null()
            };
            let err = K::kernel(&MatMatMulKerSpec {
                a: &PanelStore::Packed { ptr: pa.as_ptr() },
                b: &PanelStore::Packed { ptr: pb.as_ptr() },
                c: &mut c,
                linear: &LinearSpec::k(self.k),
                non_linear,
            });
            assert_eq!(err, 0);
            v
        }
    }

    #[derive(Debug, new)]
    pub struct PackedOffsetsProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TA, TB, TC, TI>,
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
        K: MatMatMulKer<TA, TB, TC, TI>,
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
        K: MatMatMulKer<TA, TB, TC, TI>,
        TA: 'static + Debug + AsPrimitive<TI>,
        TB: 'static + Debug + AsPrimitive<TI>,
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
            let pa = Buffer::realign_data(&self.a, K::alignment_bytes_packed_a());
            let rows_offset: Vec<isize> = self
                .rows_offsets
                .iter()
                .map(|o| (o * std::mem::size_of::<TB>()) as isize)
                .collect();
            let col_ptrs: Vec<*const TB> = unsafe {
                self.cols_offsets.iter().map(|o| self.b.as_ptr().offset(*o as isize)).collect()
            };
            let mut v = vec![TC::zero(); K::mr() * K::nr()];
            let mut c = mmm_stride_storage(&mut v, K::nr(), 1);
            let non_linear_ops = [FusedKerSpec::ScalarAdd(TI::one()), FusedKerSpec::Done];
            let non_linear = if self.add_one {
                non_linear_ops.as_ptr()
            } else {
                std::ptr::null()
            };
            let err = K::kernel(&MatMatMulKerSpec {
                a: &PanelStore::Packed { ptr: pa.as_ptr() },
                b: &PanelStore::OffsetsAndPtrs {
                    row_byte_offsets: rows_offset.as_ptr(),
                    col_ptrs: col_ptrs.as_ptr(),
                },
                c: &mut c,
                linear: &LinearSpec::k(self.rows_offsets.len()),
                non_linear,
            });
            assert_eq!(err, 0);
            v
        }
    }

    pub fn packed_packed<K, TA, TB, TC, TI>(k: usize)
    where
        K: MatMatMulKer<TA, TB, TC, TI>,
        TA: Copy + One,
        TB: Copy + One,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy + Add + Mul + Zero + Debug + fmt::Display,
        usize: AsPrimitive<TC>,
    {
        let len = K::mr() * K::nr();
        let pa = Buffer::realign_data(&vec![TA::one(); K::mr() * k], K::alignment_bytes_packed_a());
        let pb = Buffer::realign_data(&vec![TB::one(); K::nr() * k], K::alignment_bytes_packed_b());
        let mut v: Vec<TC> = vec![TC::zero(); len];
        let mut c = mmm_stride_storage(&mut v, K::nr(), 1);
        let err = K::kernel(&MatMatMulKerSpec {
            a: &PanelStore::Packed { ptr: pa.as_ptr() },
            b: &PanelStore::Packed { ptr: pb.as_ptr() },
            c: &mut c,
            linear: &LinearSpec::k(k),
            non_linear: std::ptr::null(),
        });
        assert_eq!(err, 0);
        let expected = vec![k.as_(); len];
        assert_eq!(v, expected);
    }

    pub fn mmm_stride_storage<T: Copy>(v: &mut [T], rsc: usize, csc: usize) -> PanelStore<T> {
        PanelStore::Strides {
            ptr: v.as_mut_ptr(),
            row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
            col_byte_stride: (std::mem::size_of::<T>() * csc) as isize,
            item_size: std::mem::size_of::<T>(),
        }
    }

    pub fn packed_offsets<K, TA, TB, TC, TI>(k: usize, t: usize)
    where
        K: MatMatMulKer<TA, TB, TC, TI>,
        TA: Copy + One + AsPrimitive<TI>,
        TB: Copy + One + AsPrimitive<TI>,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy + Add + Zero + Mul<Output = TI> + Debug + fmt::Display + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TA> + AsPrimitive<TB>,
    {
        let a: Vec<TA> = (1..=(k * K::mr())).map(|x| x.as_()).collect();
        let pa = Buffer::realign_data(&a, K::alignment_bytes_packed_a());
        let b: Vec<TB> = (0..(k * t)).map(|x| x.as_()).collect();
        let len = K::mr() * K::nr();
        let mut v: Vec<TC> = vec![TC::zero(); len];
        let mut c = mmm_stride_storage(&mut v, K::nr(), 1);
        let col_ptrs = (0..K::nr()).map(|i| (&b[i]) as _).collect::<Vec<_>>();
        let row_byte_offsets =
            (0..k).map(|i| (i * std::mem::size_of::<TB>() * t) as isize).collect::<Vec<_>>();
        let err = K::kernel(&MatMatMulKerSpec {
            a: &PanelStore::Packed { ptr: pa.as_ptr() },
            b: &PanelStore::OffsetsAndPtrs {
                col_ptrs: col_ptrs.as_ptr(),
                row_byte_offsets: row_byte_offsets.as_ptr(),
            },
            c: &mut c,
            linear: &LinearSpec::k(k),
            non_linear: std::ptr::null(),
        });
        assert_eq!(err, 0);
        let expected: Vec<TC> = (0..v.len())
            .map(|ix| {
                let row = ix / K::nr();
                let col = ix % K::nr();
                (0..k)
                    .map(|i| pa[K::mr() * i + row].as_() * b[t * i + col].as_())
                    .fold(TI::zero(), |s, a| s + a)
                    .as_()
            })
            .collect();
        assert_eq!(v, expected);
    }

    pub fn packed_vec<K, TA, TB, TC, TI>(k: usize)
    where
        K: MatMatMulKer<TA, TB, TC, TI>,
        TA: Copy + One + AsPrimitive<TI> + Debug,
        TB: Copy + One + AsPrimitive<TI> + Debug,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy + Add + Zero + Mul<Output = TI> + Debug + fmt::Display + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TC>,
    {
        let pa = Buffer::realign_data(&vec![TA::one(); K::mr() * k], K::alignment_bytes_packed_a());
        let b = vec![TB::one(); k];
        let c: Vec<TC> = vec![TC::zero(); K::mr()];
        let err = K::kernel(&MatMatMulKerSpec {
            a: &PanelStore::Packed { ptr: pa.as_ptr() },
            b: &PanelStore::VecStride {
                ptr: b.as_ptr(),
                byte_stride: std::mem::size_of::<TB>() as isize,
                item_size: std::mem::size_of::<TB>(),
            },
            c: &PanelStore::VecStride {
                ptr: c.as_ptr(),
                byte_stride: std::mem::size_of::<TC>() as isize,
                item_size: std::mem::size_of::<TC>(),
            },
            linear: &LinearSpec::k(k),
            non_linear: std::ptr::null(),
        });
        assert_eq!(err, 0);
        let expected = vec![k.as_(); K::mr()];
        assert_eq!(c, expected);
    }
}
