use std::marker::PhantomData;

use tract_data::TractResult;

use crate::{element_wise::ElementWiseKer, LADatum, frame::{ElementWise, element_wise_helper::map_slice_with_alignment}};

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
    fn name() -> &'static str;
    fn alignment_bytes() -> usize {
        ElementWiseKer::<T, T>::alignment_bytes()
    }
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(vec: &mut [T], scalar: T) {
        ElementWiseKer::<T, T>::run(vec, scalar)
    }
    fn by_scalar() -> Box<dyn ByScalarKer<T>> {
        Box::new(ByScalarImpl::<Self, T>::new())
    }
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
        ($cond:expr, $t: ty, $ker:ty, $reference: expr) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-25f32..25.0, 0..100), scalar in -25f32..25f32) {
                    if $cond {
                        $crate::frame::element_wise::test::test_element_wise::<$ker, $t>(&*xs, $reference, scalar).unwrap()
                    }
                }
            }
        };
    }
}
