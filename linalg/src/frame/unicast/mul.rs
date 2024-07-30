#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::unicast::UnicastKer;
    use crate::LADatum;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! unicast_mul_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(
                    (a, b) in (0..100_usize).prop_flat_map(|len| (vec![-25f32..25.0; len], vec![-25f32..25.0; len]))
                ) {
                    if $cond {
                        $crate::frame::unicast::mul::test::test_unicast_mul::<$ker, $t>(&*a, &*b).unwrap()
                    }
                }
            }

            #[test]
            fn empty() {
                if $cond {
                    $crate::frame::unicast::mul::test::test_unicast_mul::<$ker, $t>(&[], &[]).unwrap()
                }
            }
        };
    }

    pub fn test_unicast_mul<K: UnicastKer<T>, T: LADatum + Float>(a: &[f32], b: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
        T: AsPrimitive<f32>,
    {
        crate::setup_test_logger();
        let a: Vec<T> = a.iter().copied().map(|x| x.as_()).collect();
        let b: Vec<T> = b.iter().copied().map(|x| x.as_()).collect();
        crate::frame::unicast::test::test_unicast::<K, _>(&a, &b, |a, b| a * b)
    }
}
