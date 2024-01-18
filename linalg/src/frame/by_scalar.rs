#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::element_wise::ElementWiseKer;
    use crate::LADatum;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! mul_by_scalar_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-25f32..25.0, 0..100), scalar in -25f32..25f32) {
                    if $cond {
                        $crate::frame::by_scalar::test::test_mul_by_scalar::<$ker, $t>(&*xs, scalar).unwrap()
                    }
                }
            }
        };
    }

    pub fn test_mul_by_scalar<K: ElementWiseKer<T, T>, T: LADatum + Float>(
        values: &[f32],
        scalar: f32,
    ) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
        T: AsPrimitive<f32>,
    {
        crate::setup_test_logger();
        let values: Vec<T> = values.iter().copied().map(|x| x.as_()).collect();
        crate::frame::element_wise::test::test_element_wise_params::<K, T, _, T>(
            &values,
            |a| a * scalar.as_(),
            scalar.as_(),
        )
    }
}
