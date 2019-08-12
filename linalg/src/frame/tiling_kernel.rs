use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, Mul};

#[repr(C)]
#[derive(PartialEq, Copy, Clone)]
pub struct TileOpSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero + Debug,
{
    pub a: *const TileStorageSpec<T>,
    pub b: *const TileStorageSpec<T>,
    pub c: *const TileStorageSpec<T>,
    pub linear: *const LinearSpec,
    pub non_linear: *const NonLinearUSpec<T>,
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum LinearSpec {
    Mul { k: usize },
    Noop,
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum NonLinearUSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Done,
    Min(T),
    Max(T),
    AddC,
    PerRowMul(*const T),
    PerRowAdd(*const T),
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
            std::mem::size_of::<super::NonLinearUSpec<f32>>(),
            2 * std::mem::size_of::<usize>()
        )
    }

    #[macro_export]
    macro_rules! tile_kernel_tests {
        ($cond:expr, $ker:ty, $t:ty) => {
            mod kernel {
                #[allow(unused_imports)]
                use crate::frame::tiling_kernel::test;
                use crate::frame::tiling_kernel::TilingKer;

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
                fn return_c_mul_row() {
                    if $cond {
                        test::return_c_mul_row::<$ker, $t>()
                    }
                }

                #[test]
                fn return_c_add_row() {
                    if $cond {
                        test::return_c_add_row::<$ker, $t>()
                    }
                }

                #[test]
                fn packed_packed_1() {
                    if $cond {
                        test::packed_packed::<$ker, $t>(1)
                    }
                }

                #[test]
                fn packed_packed_2() {
                    if $cond {
                        test::packed_packed::<$ker, $t>(2)
                    }
                }

                #[test]
                fn packed_packed_13() {
                    if $cond {
                        test::packed_packed::<$ker, $t>(13)
                    }
                }

                #[test]
                fn packed_offsets_k1() {
                    if $cond {
                        test::packed_offsets::<$ker, $t>(1, <$ker>::nr())
                    }
                }

                #[test]
                fn packed_offsets_k2() {
                    if $cond {
                        test::packed_offsets::<$ker, $t>(2, <$ker>::nr())
                    }
                }

                #[test]
                fn packed_offsets_k13() {
                    if $cond {
                        test::packed_offsets::<$ker, $t>(13, <$ker>::nr())
                    }
                }

                #[test]
                fn packed_offsets_with_row_stride() {
                    if $cond {
                        test::packed_offsets::<$ker, $t>(2, <$ker>::nr() + 5)
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
            non_linear: std::ptr::null(),
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
            non_linear: &[NonLinearUSpec::AddC, NonLinearUSpec::Done] as _,
        });
        assert_eq!(err, 0);
        assert!(v.iter().enumerate().all(|(ix, &a)| a == (ix as f32).into()));
    }

    pub fn return_c_mul_row<K, T>()
    where
        K: TilingKer<T>,
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        let len = K::mr() * K::nr();
        let mut v: Vec<T> = (0..len).map(|f| (f as f32).into()).collect();
        let bias: Vec<T> = (0..K::mr()).map(|f| (f as f32).into()).collect();
        let mut c = tile_stride_storage(&mut v, K::nr());
        let err = K::kernel(&TileOpSpec {
            a: &null_packed_storage(),
            b: &null_packed_storage(),
            c: &mut c,
            linear: &LinearSpec::Mul { k: 0 },
            non_linear: &[
                NonLinearUSpec::AddC,
                NonLinearUSpec::PerRowMul(bias.as_ptr()),
                NonLinearUSpec::Done,
            ] as _,
        });
        assert_eq!(err, 0);
        assert!(v.iter().enumerate().all(|(ix, &a)| {
            let row = ix / K::nr();
            a == T::from(ix as f32) * bias[row]
        }));
    }

    pub fn return_c_add_row<K, T>()
    where
        K: TilingKer<T>,
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        let len = K::mr() * K::nr();
        let mut v: Vec<T> = (0..len).map(|f| (f as f32).into()).collect();
        let bias: Vec<T> = (0..K::mr()).map(|f| (f as f32).into()).collect();
        let mut c = tile_stride_storage(&mut v, K::nr());
        let err = K::kernel(&TileOpSpec {
            a: &null_packed_storage(),
            b: &null_packed_storage(),
            c: &mut c,
            linear: &LinearSpec::Mul { k: 0 },
            non_linear: &[
                NonLinearUSpec::AddC,
                NonLinearUSpec::PerRowAdd(bias.as_ptr()),
                NonLinearUSpec::Done,
            ] as _,
        });
        assert_eq!(err, 0);
        assert!(v.iter().enumerate().all(|(ix, &a)| {
            let row = ix / K::nr();
            a == T::from(ix as f32) + bias[row]
        }));
    }

    pub fn packed_packed<K, T>(k: usize)
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
            non_linear: std::ptr::null(),
        });
        assert_eq!(err, 0);
        assert!(v.iter().all(|&a| a == (k as f32).into()));
    }

    pub fn packed_offsets<K, T>(k: usize, t: usize)
    where
        K: TilingKer<T>,
        T: Mul + Add + Zero + One + Debug + Copy + PartialEq + From<f32>,
    {
        let a = (1..=(k * K::mr())).map(|x| (x as f32).into()).collect();
        let pa = realign_vec(a, K::alignment_bytes_packed_a());
        let b: Vec<T> = (0..(k * t)).map(|x| (x as f32).into()).collect();
        let len = K::mr() * K::nr();
        let mut v: Vec<T> = vec![T::zero(); len];
        let mut c = tile_stride_storage(&mut v, K::nr());
        let col_ptrs = (0..K::nr()).map(|i| (&b[i]) as _).collect::<Vec<_>>();
        let row_byte_offsets =
            (0..k).map(|i| (i * std::mem::size_of::<T>() * t) as isize).collect::<Vec<_>>();
        let err = K::kernel(&TileOpSpec {
            a: &TileStorageSpec::Packed { ptr: pa.as_ptr() },
            b: &TileStorageSpec::OffsetsAndPtrs {
                col_ptrs: col_ptrs.as_ptr(),
                row_byte_offsets: row_byte_offsets.as_ptr(),
            },
            c: &mut c,
            linear: &LinearSpec::Mul { k },
            non_linear: std::ptr::null(),
        });
        assert_eq!(err, 0);
        assert!(v.iter().enumerate().all(|(ix, &v)| {
            let row = ix / K::nr();
            let col = ix % K::nr();
            let s = (0..k)
                .map(|i| pa[K::mr() * i + row] * b[t * i + col])
                .fold(T::zero(), |s, a| s + a);
            v == s
        }));
    }
}
