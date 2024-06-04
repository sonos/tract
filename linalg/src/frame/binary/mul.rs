#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::binary::BinaryKer;
    use crate::LADatum;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! binary_mul_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        $crate::frame::binary::mul::test::test_binary_mul::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn empty() {
                if $cond {
                    $crate::frame::binary::mul::test::test_binary_mul::<$ker, $t>(&[]).unwrap()
                }
            }
        };
    }

    pub fn test_binary_mul<K: BinaryKer<T>, T: LADatum + Float>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
        T: AsPrimitive<f32>,
    {
        crate::setup_test_logger();
        let a: Vec<T> = values.iter().copied().map(|x| x.as_()).collect();
        let b: Vec<T> = values.iter().rev().copied().map(|x| x.as_()).collect();
        crate::frame::binary::test::test_binary::<K, _>(&a, &b, |a, b| a * b)
    }
}
