#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::reduce::MapReduceKer;
    use crate::LADatum;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! softmax_l2_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-25f32..25.0, 1..100)) {
                    if $cond {
                        $crate::frame::reduce::softmax::test::test_softmax_l2::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn single() {
                if $cond {
                    $crate::frame::reduce::softmax::test::test_softmax_l2::<$ker, $t>(&[0.0]).unwrap()
                }
            }

            #[test]
            fn two_zeros() {
                if $cond {
                    $crate::frame::reduce::softmax::test::test_softmax_l2::<$ker, $t>(&[0.0, 0.0]).unwrap()
                }
            }

            #[test]
            fn two_0() {
                if $cond {
                    $crate::frame::reduce::softmax::test::test_softmax_l2::<$ker, $t>(&[
                        16.62555, 21.950674,
                    ])
                    .unwrap()
                }
            }

            #[test]
            fn two_1() {
                if $cond {
                    $crate::frame::reduce::softmax::test::test_softmax_l2::<$ker, $t>(&[0.0f32, 0.38132212])
                        .unwrap()
                }
            }

            #[test]
            fn two_missing_max() {
                if $cond {
                    $crate::frame::reduce::softmax::test::test_softmax_l2::<$ker, $t>(&[
                        -46.15512, 42.875168,
                    ])
                    .unwrap()
                }
            }
        };
    }

    pub fn test_softmax_l2<K: MapReduceKer<T, T>, T>(
        values: &[f32],
    ) -> TestCaseResult
    where
        T: LADatum + Float + AsPrimitive<f32>,
        f32: AsPrimitive<T>,
    {
        use crate::generic::reduce::softmax_l2::fast_compact_exp_f32;
        crate::setup_test_logger();
        let max = values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        let values: Vec<T> = values.iter().copied().map(|x| x.as_()).collect();
        crate::frame::reduce::test::test_map_reduce_params::<K, T, T>(
            &values,
            <T as Float>::min_value(),
            T::zero(),
            //            |x| (x - max.as_()).exp(),
            |x| fast_compact_exp_f32(x.as_() - max).as_(),
            |a, b| a + b,
            max.as_(),
        )
    }
}
