use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, Mul};

#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct TileOpSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    pub a: *const TileStorageSpec<T>,
    pub b: *const TileStorageSpec<T>,
    pub c: *const TileStorageSpec<T>,
    pub linear: *const LinearSpec,
    pub non_linear: *const NonLinearSpec<T>,
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum LinearSpec {
    Mul { k: usize },
    Noop,
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum NonLinearSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Done,
    Min(T),
    Max(T),
    AddC,
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone)]
pub enum TileStorageSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Strides { ptr: *mut T, row_byte_stride: isize, col_byte_stride: isize },
    Packed { ptr: *const T },
    OffsetsAndPtrs { row_byte_offsets: *const isize, col_ptrs: *const *const T },
}

pub trait TilingKer<T>: Copy + Clone + Debug + Send + Sync
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    #[inline(always)]
    fn name() -> &'static str;
    #[inline(always)]
    fn kernel(op: &TileOpSpec<T>) -> isize;
    #[inline(always)]
    fn mr() -> usize;
    #[inline(always)]
    fn nr() -> usize;
    #[inline(always)]
    fn alignment_bytes_packed_a() -> usize;
    #[inline(always)]
    fn alignment_bytes_packed_b() -> usize;
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::align::realign_vec;
    use num_traits::One;

    #[test]
    fn check_non_linear_enum_size() {
        assert_eq!(
            std::mem::size_of::<super::NonLinearSpec<f32>>(),
            2 * std::mem::size_of::<usize>()
        )
    }

    #[macro_export]
    macro_rules! tile_kernel_tests {
        ($cond:expr, $ker:ty, $t:ty) => {
            mod kernel {
                #[allow(unused_imports)]
                use crate::frame::tiling::test::*;
                use crate::frame::tiling_kernel::*;
                use crate::frame::tiling_kernel::test;
                /*
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
                            let found = pb.run::<$ker>();
                            let expected = pb.expected();
                            proptest::prop_assert_eq!(found, expected)
                        }
                    }
                }
                */

                fn packed(v: &mut [$t]) -> TileStorageSpec<$t> {
                    TileStorageSpec::Packed { ptr: v.as_mut_ptr() }
                }

                #[test]
                fn return_zeros() {
                    if $cond {
                        test::return_zeros::<$ker, $t>()
                    }
                }

                #[test]
                fn return_c() {
                    if $cond {
                        test::return_c::<$ker, $t>()
                    }
                }

                #[test]
                fn return_mul_k_1() {
                    if $cond {
                        test::mul_k::<$ker, $t>(1)
                    }
                }
            }
        };
    }

    pub fn null_packed_storage<T>() -> TileStorageSpec<T>
    where
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        TileStorageSpec::Packed { ptr: std::ptr::null::<T>() as _ }
    }

    pub fn null_stride_storage<T>() -> TileStorageSpec<T>
    where
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        TileStorageSpec::Strides {
            ptr: std::ptr::null::<T>() as _,
            row_byte_stride: 0,
            col_byte_stride: 0,
        }
    }

    pub fn tile_stride_storage<T>(v: &mut [T], rsc: usize) -> TileStorageSpec<T>
    where
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        TileStorageSpec::Strides {
            ptr: v.as_mut_ptr(),
            row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
            col_byte_stride: std::mem::size_of::<T>() as isize,
        }
    }

    pub fn return_zeros<K, T>()
    where
        K: TilingKer<T>,
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        let mut v = vec![99f32.into(); K::mr() * K::nr()];
        let mut c = tile_stride_storage(&mut v, K::nr());
        let err = K::kernel(&TileOpSpec {
            a: &null_packed_storage(),
            b: &null_packed_storage(),
            c: &mut c,
            linear: &LinearSpec::Mul { k: 0 },
            non_linear: &NonLinearSpec::Done,
        });
        assert_eq!(err, 0);
        assert!(v.iter().all(|&a| a.is_zero()));
    }

    pub fn return_c<K, T>()
    where
        K: TilingKer<T>,
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        let len = K::mr() * K::nr();
        let mut v: Vec<T> = (0..len).map(|f| (f as f32).into()).collect();
        let mut c = tile_stride_storage(&mut v, K::nr());
        let err = K::kernel(&TileOpSpec {
            a: &null_packed_storage(),
            b: &null_packed_storage(),
            c: &mut c,
            linear: &LinearSpec::Mul { k: 0 },
            non_linear: &[NonLinearSpec::AddC, NonLinearSpec::Done] as _,
        });
        assert_eq!(err, 0);
        assert!(v.iter().enumerate().all(|(ix, &a)| a == (ix as f32).into()));
    }

    pub fn mul_k<K, T>(k: usize)
    where
        K: TilingKer<T>,
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        let len = K::mr() * K::nr();
        let pa = realign_vec(vec![T::one(); K::mr() * k], K::alignment_bytes_packed_a());
        let pb = realign_vec(vec![T::one(); K::nr() * k], K::alignment_bytes_packed_b());
        let mut v: Vec<T> = vec![T::zero(); len];
        let mut c = tile_stride_storage(&mut v, K::nr());
        let err = K::kernel(&TileOpSpec {
            a: &TileStorageSpec::Packed { ptr: pa.as_ptr() },
            b: &TileStorageSpec::Packed { ptr: pb.as_ptr() },
            c: &mut c,
            linear: &LinearSpec::Mul { k },
            non_linear: &[NonLinearSpec::AddC, NonLinearSpec::Done] as _,
        });
        assert_eq!(err, 0);
        assert!(v.iter().all(|&a| a == (k as f32).into()));
    }
}
