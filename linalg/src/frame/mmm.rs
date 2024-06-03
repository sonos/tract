pub mod cost_model;
#[macro_use]
pub(crate) mod fuse;
#[macro_use]
pub(crate) mod kernel;
pub(crate) mod input_store;
#[macro_use]
#[allow(clippy::module_inception)]
pub(crate) mod mmm;
mod scratch;
mod storage;
#[cfg(test)]
#[macro_use]
pub mod tests;

pub use cost_model::*;
pub use fuse::*;
pub use input_store::*;
pub use kernel::MatMatMulKer;
pub use mmm::*;
pub use scratch::*;
pub use storage::*;

pub fn no_prefetch(_ptr: *const u8, _len: usize) {}

macro_rules! MMMKernel {
    ($ti:ident, $func:ident; $mr: expr, $nr: expr; $alignment_bytes_packed_a: expr, $alignment_bytes_packed_b: expr; $end_padding_packed_a: expr, $end_padding_packed_b: expr ; $prefetch: ident, $cond: expr $(, can_fuse: $can_fuse:expr)?) => {
        paste! {
            mod [<sys_ $func>] {
                use crate::frame::mmm::*;
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                extern_kernel!(fn $func(op: *const FusedKerSpec<$ti>) -> isize);
            }

            #[allow(non_camel_case_types)]
            #[derive(Copy, Clone, Debug, new)]
            pub struct $func;

            impl $crate::frame::mmm::MatMatMulKer for $func {
                type Acc = $ti;
                #[inline(always)]
                fn name() -> &'static str {
                    stringify!($func)
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
                fn kernel(spec: &[$crate::frame::mmm::FusedKerSpec<$ti>]) -> isize {
                    debug_assert!(spec.len() > 0);
                    debug_assert!(matches!(spec[spec.len() - 1], $crate::frame::mmm::FusedKerSpec::Done));
                    unsafe { [<sys_ $func>]::$func(spec.as_ptr()) }
                }
                #[inline(always)]
                fn prefetch(ptr: *const u8, len: usize) {
                    ($prefetch)(ptr, len)
                }
                $(
                    fn can_fuse(spec: &FusedSpec) -> bool {
                        ($can_fuse)(spec)
                    }
                )?
            }
        }
        test_mmm_kernel!($ti, $func, $cond);
    };
}

macro_rules! test_mmm_kernel {
    (f16, $func:ident, $cond: expr) => {
        test_mmm_kernel_f16!($func, $cond);
    };
    (f32, $func:ident, $cond: expr) => {
        test_mmm_kernel_f32!($func, $cond);
    };
    (i32, $func:ident, $cond: expr) => {
        test_mmm_kernel_i32!($func, $cond);
    };
}
