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
            mmm_frame_tests!($cond, $k, i8, i8, i8, i32);
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
            mmm_frame_tests!($cond, $k, i8, i8, i32, i32);
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
    use std::fmt;
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
                #[allow(unused_imports)]
                use crate::frame::mmm::kernel::test;
                use crate::frame::mmm::MatMatMulKer;

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
        let mut c = mmm_stride_storage(&mut v, K::nr());
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

    pub fn mmm_stride_storage<T: Copy>(v: &mut [T], rsc: usize) -> PanelStore<T> {
        PanelStore::Strides {
            ptr: v.as_mut_ptr(),
            row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
            col_byte_stride: std::mem::size_of::<T>() as isize,
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
        let mut c = mmm_stride_storage(&mut v, K::nr());
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
            },
            c: &PanelStore::VecStride {
                ptr: c.as_ptr(),
                byte_stride: std::mem::size_of::<TC>() as isize,
            },
            linear: &LinearSpec::k(k),
            non_linear: std::ptr::null(),
        });
        assert_eq!(err, 0);
        let expected = vec![k.as_(); K::mr()];
        assert_eq!(c, expected);
    }
}
