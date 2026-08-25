use std::fmt::Debug;
use std::marker::PhantomData;

use crate::element_wise::{ElementWise, ElementWiseKer};
use crate::element_wise_helper::map_slice_with_alignment;
use crate::{BinFn, LADatum};
use tract_data::internal::*;

/// Generic implementation struct that unify all by scalar kernels.
/// A by scalar operation is an ElementWise operation with a scalar paramerer.
#[derive(Debug, Clone, new)]
pub struct ByScalarImpl<K, T>
where
    T: LADatum,
    K: ByScalarKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> ElementWise<T, T> for ByScalarImpl<K, T>
where
    T: LADatum,
    K: ByScalarKer<T> + Clone,
{
    fn name(&self) -> &'static str {
        K::name()
    }
    fn run_with_params(&self, vec: &mut [T], params: T) -> TractResult<()> {
        map_slice_with_alignment(vec, |data| K::run(data, params), K::nr(), K::alignment_bytes())
    }
}

pub trait ByScalarKer<T>: ElementWiseKer<T, T>
where
    T: LADatum,
{
    fn bin() -> Box<BinFn> {
        Box::new(|a: &mut TensorView, b: &TensorView| {
            let a_slice = a.as_slice_mut()?;
            let b = b.as_slice()?[0];
            (Self::ew()).run_with_params(a_slice, b)
        })
    }
}

// A by-scalar binary kernel from a `run` body. A leading arch ident is for bodies that are
// inline arch asm or intrinsics, which will not even compile elsewhere: those builds get a
// signature-matched panic stub instead, so the kernel struct exists everywhere.
/// Declare a by-scalar routine: the kernel, its registry descriptors and its accuracy tests, from
/// one statement. `op` is what the kernel computes, which is what the tests compare against; `bin`
/// and `param` ask for the two shapes a by-scalar kernel can answer in -- two views, or a typed
/// scalar -- and `isa` says which machines may run it, and therefore which may test it.
macro_rules! routine_by_scalar_rust {
    (arm; $($rest:tt)*) => { routine_by_scalar_rust!(@ arm, target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => {
        routine_by_scalar_rust!(@ aarch64, target_arch = "aarch64"; $($rest)*);
    };
    (x86_64; $($rest:tt)*) => {
        routine_by_scalar_rust!(@ x86_64, target_arch = "x86_64"; $($rest)*);
    };
    (riscv64; $($rest:tt)*) => {
        routine_by_scalar_rust!(@ riscv64, target_arch = "riscv64"; $($rest)*);
    };
    (wasm32; $($rest:tt)*) => {
        routine_by_scalar_rust!(@ wasm32,
            all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*);
    };
    (portable; $($rest:tt)*) => { routine_by_scalar_rust!(@ portable, all(); $($rest)*); };

    // Each descriptor clause adds its own and hands the rest on, so the kernel and its tests are
    // written once and no clause has to be spelled inside another.
    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, op($op:ident), bin $(, param($param:ident))? $(, isa($($isa:ident),+))?) => {
        paste! {
            submit_routine!($arch; [<Bin $ti:upper>], BinByScalar($op), $ker $(, isa($($isa),+))?);
        }
        routine_by_scalar_rust!(@ $arch, $built; $ti, $ker, $nr, $alignment_items, $run,
            op($op) $(, param($param))? $(, isa($($isa),+))?);
    };

    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, op($op:ident), param($param:ident) $(, isa($($isa:ident),+))?) => {
        paste! {
            submit_routine!($arch; [<$ti:upper Param>], $param, $ker $(, isa($($isa),+))?);
        }
        routine_by_scalar_rust!(@ $arch, $built; $ti, $ker, $nr, $alignment_items, $run,
            op($op) $(, isa($($isa),+))?);
    };

    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, op($op:ident) $(, isa($($isa:ident),+))?) => {
        by_scalar_kernel!(@ $built; $ti, $ker, $nr, $alignment_items, $ti, $run);
        paste! {
            #[cfg(test)]
            mod [<test_ $ker:snake>] {
                use super::*;
                by_scalar_frame_tests!(
                    cfg!($built)
                        && $crate::isa::IsaReq::ANY
                            $(.needing(&[$($crate::isa::Isa::$isa),+]))?
                            .satisfied_by($crate::isa::native()),
                    $ti,
                    $ker,
                    bin_reference!($op)
                );
            }
        }
    };
}

macro_rules! by_scalar_kernel {
    (arm; $($rest:tt)*) => { by_scalar_kernel!(@ target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { by_scalar_kernel!(@ target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { by_scalar_kernel!(@ target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { by_scalar_kernel!(@ target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => { by_scalar_kernel!(@ all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $built:meta; $ti:ident, $func:ident, $nr:expr, $alignment_items:expr, $params:ty, $run:item) => {
        #[cfg($built)]
        by_scalar_kernel!($ti, $func, $nr, $alignment_items, $params, $run);
        #[cfg(not($built))]
        by_scalar_kernel!($ti, $func, $nr, $alignment_items, $params,
            fn run(_vec: &mut [$ti], _params: $params) {
                panic!(concat!(stringify!($func), ": kernel not built for this target"))
            }
        );
    };

    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $params: ty, $run: item) => {
        paste! {
            ew_kernel!($ti, $func, $nr, $alignment_items, $ti, $run);

            impl crate::frame::by_scalar::ByScalarKer<$ti> for $func {}
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::LADatum;
    use crate::frame::element_wise::ElementWiseKer;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! by_scalar_frame_tests {
        ($cond:expr, $t: ty, $ker:ty, $func:expr) => {
            pastey::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<prop_ $ker:snake>](xs in proptest::collection::vec(-25f32..25.0, 0..100), scalar in -25f32..25f32) {
                        if $cond {
                            $crate::frame::by_scalar::test::test_by_scalar::<$ker, $t>(&*xs, scalar, $func).unwrap()
                        }
                    }
                }
            }
        };
    }

    pub fn test_by_scalar<K: ElementWiseKer<T, T>, T: LADatum + Float>(
        values: &[f32],
        scalar: f32,
        func: impl Fn(T, T) -> T,
    ) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::setup_test_logger();
        let values: Vec<T> = values.iter().copied().map(|x| x.as_()).collect();
        crate::frame::element_wise::test::test_element_wise_params::<K, T, _, T>(
            &values,
            |a| (func)(a, scalar.as_()),
            scalar.as_(),
        )
    }
}
