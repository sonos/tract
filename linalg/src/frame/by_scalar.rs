use std::fmt::Debug;
use std::marker::PhantomData;

use crate::element_wise::{ElementWise, ElementWiseKer};
use crate::element_wise_helper::map_slice_with_alignment;
use crate::{LADatum, LinalgFn};
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
    fn bin() -> Box<LinalgFn> {
        Box::new(|a: &mut TensorView, b: &TensorView| {
            let a_slice = a.as_slice_mut()?;
            let b = b.as_slice()?[0];
            (Self::ew()).run_with_params(a_slice, b)
        })
    }
}

macro_rules! by_scalar_impl_wrap {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $params: ty, $run: item) => {
        paste! {
            ew_impl_wrap!($ti, $func, $nr, $alignment_items, $ti, $run);

            impl crate::frame::by_scalar::ByScalarKer<$ti> for $func {}
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::element_wise::ElementWiseKer;
    use crate::LADatum;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! by_scalar_frame_tests {
        ($cond:expr, $t: ty, $ker:ty, $func:expr) => {
            paste::paste! {
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
