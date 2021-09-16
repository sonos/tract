#[macro_use]
pub(crate) mod fuse;
#[macro_use]
pub(crate) mod kernel;
#[macro_use]
pub(crate) mod mmm;
mod scratch;
mod storage;
#[cfg(test)]
#[macro_use]
pub mod tests;

pub use fuse::*;
pub use kernel::*;
pub use mmm::*;
pub use scratch::*;
pub use storage::*;

#[cfg(test)]
pub use tests::*;

macro_rules! MMMKernel {
    ($typ:ident<$ti:ident>, $name:expr, $func:ident; $mr: expr, $nr: expr; $alignment_bytes_packed_a: expr, $alignment_bytes_packed_b: expr; $end_padding_packed_a: expr, $end_padding_packed_b: expr $(, $prefetch: ident)?) => {
        #[derive(Copy, Clone, Debug, new)]
        pub struct $typ;

        impl MatMatMulKer<$ti> for $typ {
            #[inline(always)]
            fn name() -> &'static str {
                $name
            }
            #[inline(always)]
            fn mr() -> usize {
                $mr
            }
            #[inline(always)]
            fn nr() -> usize {
                $nr
            }
            #[inline(always)]
            fn alignment_bytes_packed_a() -> usize {
                $alignment_bytes_packed_a
            }
            #[inline(always)]
            fn alignment_bytes_packed_b() -> usize {
                $alignment_bytes_packed_b
            }
            #[inline(always)]
            fn end_padding_packed_a() -> usize {
                $end_padding_packed_a
            }
            #[inline(always)]
            fn end_padding_packed_b() -> usize {
                $end_padding_packed_b
            }
            #[inline(always)]
            fn kernel(spec: &[FusedKerSpec<$ti>]) -> isize {
                debug_assert!(spec.len() > 0);
                debug_assert!(matches!(spec[spec.len() - 1], FusedKerSpec::Done));
                unsafe { $func(spec.as_ptr()) }
            }
            $(
                fn prefetch(ptr: *const u8, len: usize) {
                    ($prefetch)(ptr, len)
                }
            )?
        }
    };
}
