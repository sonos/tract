#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::reduce::ReduceKer;
    use crate::LADatum;
    use num_traits::{AsPrimitive, Float, Zero};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! sum_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-25_isize..25, 0..100)) {
                    if $cond {
                        let xs_float = xs.into_iter().map(|it| it as f32).collect::<Vec<_>>();
                        $crate::frame::reduce::sum::test::test_sum::<$ker, $t>(&*xs_float).unwrap()
                    }
                }
            }

            #[test]
            fn empty() {
                if $cond {
                    $crate::frame::reduce::sum::test::test_sum::<$ker, $t>(&[]).unwrap()
                }
            }

            #[test]
            fn simple() {
                if $cond {
                    $crate::frame::reduce::sum::test::test_sum::<$ker, $t>(&[1.0, 2.0]).unwrap()
                }
            }
            #[test]
            fn multiple_tile() {
                if $cond {
                    $crate::frame::reduce::sum::test::test_sum::<$ker, $t>(&[1.0; 35]).unwrap()
                }
            }
        };
    }

    pub fn test_sum<K: ReduceKer<T>, T: LADatum + Float + Zero>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
        T: AsPrimitive<f32>,
    {
        crate::setup_test_logger();
        let values: Vec<T> = values.iter().copied().map(|x| x.as_()).collect();
        crate::frame::reduce::test::test_reduce::<K, _>(&values, <T as Zero>::zero(), |a, b| a + b)
    }
}
