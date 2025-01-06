#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::reduce::ReduceKer;
    use crate::LADatum;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! max_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        $crate::frame::reduce::max::test::test_max::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn empty() {
                if $cond {
                    $crate::frame::reduce::max::test::test_max::<$ker, $t>(&[]).unwrap()
                }
            }
        };
    }

    pub fn test_max<K: ReduceKer<T>, T: LADatum + Float>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::setup_test_logger();
        let values: Vec<T> = values.iter().copied().map(|x| x.as_()).collect();
        crate::frame::reduce::test::test_reduce::<K, _>(
            &values,
            <T as Float>::min_value(),
            |a, b| a.max(b),
        )
    }
}
