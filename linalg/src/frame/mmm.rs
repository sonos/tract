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
    ($typ:ident<$ti:ident>, $name:expr, $func:ident; $mr: expr, $nr: expr; $alignment_bytes_packed_a: expr, $alignment_bytes_packed_b: expr; $end_padding_packed_a: expr, $end_padding_packed_b: expr) => {
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
            fn alignment_bytes_packed_a() -> usize {
                $alignment_bytes_packed_a
            }
            fn alignment_bytes_packed_b() -> usize {
                $alignment_bytes_packed_b
            }
            fn end_padding_packed_a() -> usize {
                $end_padding_packed_a
            }
            fn end_padding_packed_b() -> usize {
                $end_padding_packed_b
            }
            #[inline(never)]
            fn kernel(spec: &MatMatMulKerSpec<$ti>) -> isize {
                unsafe { $func(spec) }
            }
        }
    }
}

