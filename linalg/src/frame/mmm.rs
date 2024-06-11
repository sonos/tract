pub mod cost_model;
#[macro_use]
pub(crate) mod fuse;
#[macro_use]
pub(crate) mod kernel;
pub(crate) mod input_store;
#[macro_use]
#[allow(clippy::module_inception)]
pub(crate) mod mmm;
pub mod pack;
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

macro_rules! MMMExternKernel {
    ($ti:ident, $func:ident; $mr: expr, $nr: expr; $alignment_bytes_packed_a: expr, $alignment_bytes_packed_b: expr; $end_padding_packed_a: expr, $end_padding_packed_b: expr ; $prefetch: path, $cond: expr $(, can_fuse: $can_fuse:expr)?) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use super::*;
                #[allow(unused_imports)]
                use crate::frame::mmm::*;
                extern_kernel!(fn $func(op: *const FusedKerSpec<$ti>) -> isize);
            }
            MMMKernelWrapper!($ti, $func; [<sys_ $func>]::$func; $mr, $nr; $alignment_bytes_packed_a, $alignment_bytes_packed_b; $end_padding_packed_a, $end_padding_packed_b; $prefetch, $cond $(, can_fuse: $can_fuse)?);
        }
    }
}

macro_rules! MMMKernelWrapper {
    (   $ti:ident, $id:ident;
        $func: path;
        $mr: expr, $nr: expr;
        $alignment_bytes_packed_a: expr, $alignment_bytes_packed_b: expr;
        $end_padding_packed_a: expr, $end_padding_packed_b: expr ;
        $prefetch: path, $cond: expr
        $(, can_fuse: $can_fuse:expr)?
        $(, packing_defs: { $($packing_def: item)* })?
        $(, packings: $($packing: ident)*)?
        ) => {

        paste! {
            #[allow(non_camel_case_types)]
            #[derive(Copy, Clone, Debug, new, Default)]
            pub struct [<$id:camel>];

            impl [<$id:camel>] {
                pub fn mmm(&self) -> Box<dyn $crate::frame::mmm::MatMatMul> {
                    Box::new(Self)
                }
            }

            mod [<packing_ $id>] {
                use $crate::frame::mmm::pack::Packer;
                use $crate::frame::mmm::MMMInputFormat;
                use tract_data::prelude::*;

                const NATIVE_A: Packer = Packer::new(DatumType::[<$ti:upper>], $mr, $alignment_bytes_packed_a, $end_padding_packed_a);
                const NATIVE_B: Packer = Packer::new(DatumType::[<$ti:upper>], $nr, $alignment_bytes_packed_b, $end_padding_packed_b);
                const NATIVE: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&NATIVE_A, &NATIVE_B);
                $($($packing_def)*)?
                pub const PACKINGS: &[(&dyn MMMInputFormat, &dyn MMMInputFormat)] = &[NATIVE $( $(, $packing)* )?];
            }

            impl $crate::frame::mmm::MatMatMulKer for  [<$id:camel>] {
                type Acc = $ti;
                #[inline(always)]
                fn name(&self) -> std::borrow::Cow<'static, str> {
                    std::borrow::Cow::Borrowed(stringify!($id))
                }
                #[inline(always)]
                fn mr(&self) -> usize {
                    $mr
                }
                #[inline(always)]
                fn nr(&self) -> usize {
                    $nr
                }
                #[inline(always)]
                fn packings(&self) -> &[(&dyn crate::frame::mmm::MMMInputFormat, &dyn crate::frame::mmm::MMMInputFormat)] {
                    &[<packing_ $id>]::PACKINGS
                }
                #[inline(always)]
                fn kernel(&self, spec: &[$crate::frame::mmm::FusedKerSpec<$ti>]) -> isize {
                    debug_assert!(spec.len() > 0);
                    debug_assert!(matches!(spec[spec.len() - 1], $crate::frame::mmm::FusedKerSpec::Done));
                    unsafe { $func(spec.as_ptr()) }
                }
                #[inline(always)]
                fn prefetch(&self, ptr: *const u8, len: usize) {
                    ($prefetch)(ptr, len)
                }
                $(
                    fn can_fuse(&self, spec: &FusedSpec) -> bool {
                        ($can_fuse)(spec)
                    }
                 )?
            }

            #[allow(non_upper_case_globals)]
            pub const $id: [<$id:camel>] = [<$id:camel>];
        }
        test_mmm_kernel!($ti, $id, $cond);
    }
}

macro_rules! test_mmm_kernel {
    (f16, $func:ident, $cond: expr) => {
        test_mmm_kernel_f16!($func, $cond);
    };
    (f32, $func:ident, $cond: expr) => {
        test_mmm_kernel_f32!($func, $cond);
    };
    (f64, $func:ident, $cond: expr) => {
        test_mmm_kernel_f64!($func, $cond);
    };
    (i32, $func:ident, $cond: expr) => {
        test_mmm_kernel_i32!($func, $cond);
    };
}
