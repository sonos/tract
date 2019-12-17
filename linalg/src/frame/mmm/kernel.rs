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

pub trait MatMatMulKer<TA, TB, TC, TI>: Copy + Clone + Debug + Send + Sync
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
            qmmm_kernel_fuse_tests!($cond, $k, i8, i8, i8, i32);
            qmmm_frame_tests!($cond, $k, i8, i8, i8, i32);
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
    use proptest::arbitrary::Arbitrary;
    use proptest::prelude::*;
    use std::fmt;
    use std::marker::PhantomData;
    use std::ops::{Add, AddAssign, Mul};

    #[test]
    fn check_non_linear_enum_size() {
        assert_eq!(
            std::mem::size_of::<super::FusedKerSpec<f32>>(),
            3 * std::mem::size_of::<usize>()
        )
    }

    #[derive(Clone, Debug)]
    pub struct PackedPackedKerProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TA, TB, TC, TI>,
        TA: Copy + One + Debug + Arbitrary + AsPrimitive<TI>,
        TB: Copy + One + Debug + Arbitrary + AsPrimitive<TI>,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy
            + Add
            + AddAssign
            + Mul
            + Zero
            + Debug
            + fmt::Display
            + AsPrimitive<TC>
            + Mul<Output = TI>,
        isize: AsPrimitive<TA> + AsPrimitive<TB> + AsPrimitive<TC>,
    {
        k: usize,
        a: Vec<TA>,
        b: Vec<TB>,
        _boo: PhantomData<(K, TC, TI)>,
    }

    impl<K, TA, TB, TC, TI> Arbitrary for PackedPackedKerProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TA, TB, TC, TI>,
        TA: Copy + One + Debug + Arbitrary + AsPrimitive<TI>,
        TB: Copy + One + Debug + Arbitrary + AsPrimitive<TI>,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy
            + Add
            + AddAssign
            + Mul
            + Zero
            + Debug
            + fmt::Display
            + AsPrimitive<TC>
            + Mul<Output = TI>,
        isize: AsPrimitive<TA> + AsPrimitive<TB> + AsPrimitive<TC>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (1usize..100)
                .prop_flat_map(|k| {
                    use proptest::collection::vec;
                    (
                        Just(k),
                        vec((-10isize..10).prop_map(|x| x.as_()), k * K::mr()..=k * K::mr()),
                        vec((-10isize..10).prop_map(|x| x.as_()), k * K::nr()..=k * K::nr()),
                    )
                })
                .prop_map(|(k, a, b)| PackedPackedKerProblem { k, a, b, _boo: PhantomData })
                .boxed()
        }
    }

    impl<K, TA, TB, TC, TI> PackedPackedKerProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TA, TB, TC, TI>,
        TA: Copy + One + Debug + Arbitrary + AsPrimitive<TI>,
        TB: Copy + One + Debug + Arbitrary + AsPrimitive<TI>,
        TC: Copy + PartialEq + Zero + 'static + Debug,
        TI: Copy
            + Add
            + AddAssign
            + Mul
            + Zero
            + Debug
            + fmt::Display
            + AsPrimitive<TC>
            + Mul<Output = TI>,
        isize: AsPrimitive<TA> + AsPrimitive<TB> + AsPrimitive<TC>,
    {
        pub fn mat(&self) -> Vec<TC> {
            let mut i = vec![TI::zero(); K::mr() * K::nr()];
            for k in 0..self.k {
                for n in 0..K::nr() {
                    for m in 0..K::mr() {
                        let a = self.a[k + self.k * m];
                        let b = self.b[n + K::nr() * k];
                        i[n + K::nr() * m] += a.as_() * b.as_()
                    }
                }
            }
            i.iter().map(|i| i.as_()).collect()
        }

        pub fn vec(&self) -> Vec<TC> {
            let mut i = vec![TI::zero(); K::mr()];
            for k in 0..self.k {
                for m in 0..K::mr() {
                    let a = self.a[k + self.k * m];
                    let b = self.b[K::nr() * k];
                    i[m] += a.as_() * b.as_()
                }
            }
            i.iter().map(|i| i.as_()).collect()
        }

        fn packed_a(&self) -> Buffer<TA> {
            let pa =
                crate::frame::PackA::new(self.k, K::mr(), K::mr(), K::alignment_bytes_packed_a());
            let mut a = Buffer::uninitialized(pa.len(), pa.alignment());
            pa.pack(a.as_mut_ptr(), self.a.as_ptr(), self.k as isize, 1);
            a
        }

        fn packed_b(&self) -> Buffer<TB> {
            let pb =
                crate::frame::PackB::new(self.k, K::nr(), K::nr(), K::alignment_bytes_packed_b());
            let mut b = Buffer::uninitialized(pb.len(), pb.alignment());
            pb.pack(b.as_mut_ptr(), self.b.as_ptr(), K::nr() as isize, 1);
            b
        }

        pub fn ones(k: usize) -> Self {
            PackedPackedKerProblem::<K, TA, TB, TC, TI> {
                k,
                a: vec![TA::one(); K::mr() * k],
                b: vec![TB::one(); K::nr() * k],
                _boo: PhantomData,
            }
        }

        pub fn counting(k: usize) -> Self {
            PackedPackedKerProblem::<K, TA, TB, TC, TI> {
                k,
                a: (0..).map(|i| i.as_()).take(K::mr() * k).collect(),
                b: (0..).map(|i| i.as_()).take(K::nr() * k).collect(),
                _boo: PhantomData,
            }
        }

        pub fn packed_packed(&self) -> Vec<TC> {
            let a = self.packed_a();
            let b = self.packed_b();
            let mut v = vec![TC::zero(); K::mr() * K::nr()];
            let ref mut c = mmm_stride_storage(&mut v, K::nr());
            let err = K::kernel(&MatMatMulKerSpec {
                a: &PanelStore::Packed { ptr: a.as_ptr() },
                b: &PanelStore::Packed { ptr: b.as_ptr() },
                c,
                linear: &LinearSpec::k(self.k),
                non_linear: std::ptr::null(),
            });
            assert_eq!(err, 0);
            v
        }

        pub fn packed_offsets(&self) -> Vec<TC> {
            let a = self.packed_a();

            let mut v = vec![TC::zero(); K::mr() * K::nr()];
            let ref mut c = mmm_stride_storage(&mut v, K::nr());

            let col_ptrs = (0..K::nr()).map(|i| (&self.b[i]) as _).collect::<Vec<_>>();
            let row_byte_offsets = (0..self.k)
                .map(|i| (i * std::mem::size_of::<TB>() * K::nr()) as isize)
                .collect::<Vec<_>>();
            let err = K::kernel(&MatMatMulKerSpec {
                a: &PanelStore::Packed { ptr: a.as_ptr() },
                b: &PanelStore::OffsetsAndPtrs {
                    col_ptrs: col_ptrs.as_ptr(),
                    row_byte_offsets: row_byte_offsets.as_ptr(),
                },
                c,
                linear: &LinearSpec::k(self.k),
                non_linear: std::ptr::null(),
            });
            assert_eq!(err, 0);
            v
        }

        pub fn packed_vec(&self) -> Vec<TC> {
            let a = self.packed_a();

            let c: Vec<TC> = vec![(-1).as_(); K::mr()];
            let err = K::kernel(&MatMatMulKerSpec {
                a: &PanelStore::Packed { ptr: a.as_ptr() },
                b: &PanelStore::VecStride {
                    ptr: self.b.as_ptr(),
                    byte_stride: (std::mem::size_of::<TB>() * K::nr()) as isize,
                },
                c: &PanelStore::VecStride {
                    ptr: c.as_ptr(),
                    byte_stride: std::mem::size_of::<TC>() as isize,
                },
                linear: &LinearSpec::k(self.k),
                non_linear: std::ptr::null(),
            });
            assert_eq!(err, 0);
            c
        }
    }

    #[macro_export]
    macro_rules! mmm_kernel_tests {
        ($cond:expr, $ker:ty, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
            mod kernel {
                #[allow(unused_imports)]
                use crate::frame::mmm::kernel::test;
                use proptest::prelude::*;
                use crate::frame::mmm::kernel::test::PackedPackedKerProblem;

                proptest::proptest! {
                    #[test]
                    fn packed_packed_prop(pb in any::<PackedPackedKerProblem<$ker, $ta, $tb, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.packed_packed(), pb.mat());
                        }
                    }

                    #[test]
                    fn packed_offsets_prop(pb in any::<PackedPackedKerProblem<$ker, $ta, $tb, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.packed_offsets(), pb.mat());
                        }
                    }
                }

                #[test]
                fn packed_packed_1() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::ones(1);
                        assert_eq!(pb.packed_packed(), pb.mat());
                    }
                }

                #[test]
                fn packed_packed_2() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::ones(2);
                        assert_eq!(pb.packed_packed(), pb.mat());
                    }
                }

                #[test]
                fn packed_packed_13() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::ones(13);
                        assert_eq!(pb.packed_packed(), pb.mat());
                    }
                }

                #[test]
                fn packed_offsets_k1() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::counting(1);
                        assert_eq!(pb.packed_offsets(), pb.mat());
                    }
                }

                #[test]
                fn packed_offsets_k2() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::counting(2);
                        assert_eq!(pb.packed_offsets(), pb.mat());
                    }
                }

                #[test]
                fn packed_offsets_k13() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::counting(13);
                        assert_eq!(pb.packed_offsets(), pb.mat());
                    }
                }

                #[test]
                fn packed_vec_k1() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::counting(1);
                        assert_eq!(pb.packed_vec(), pb.vec());
                    }
                }

                #[test]
                fn packed_vec_k2() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::counting(2);
                        assert_eq!(pb.packed_vec(), pb.vec());
                    }
                }

                #[test]
                fn packed_vec_k13() {
                    if $cond {
                        let pb = PackedPackedKerProblem::<$ker, $ta, $tb, $tc, $ti>::counting(2);
                        assert_eq!(pb.packed_vec(), pb.vec());
                    }
                }
            }
        };
    }

    pub fn mmm_stride_storage<T: Copy>(v: &mut [T], rsc: usize) -> PanelStore<T> {
        PanelStore::Strides {
            ptr: v.as_mut_ptr(),
            row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
            col_byte_stride: std::mem::size_of::<T>() as isize,
        }
    }

}
