#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::element_wise::*;
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! tanh_frame_tests {
        ($cond:expr, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn tanh(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        crate::frame::tanh::test::test_tanh::<$ker>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn tanh_4_magic() {
                if $cond {
                    crate::frame::tanh::test::test_tanh::<$ker>(&[0f32, -20.0, 20.0, 0.0]).unwrap()
                }
            }

            #[test]
            fn tanh_4zeros() {
                if $cond {
                    crate::frame::tanh::test::test_tanh::<$ker>(&[0.0; 4]).unwrap();
                }
            }

            #[test]
            fn tanh_20_ones() {
                if $cond {
                    crate::frame::tanh::test::test_tanh::<$ker>(&[1.0; 20]).unwrap();
                }
            }

            #[test]
            fn tanh_18_zeros() {
                if $cond {
                    crate::frame::tanh::test::test_tanh::<$ker>(&[0.0; 18]).unwrap();
                }
            }
        };
    }

    pub fn test_tanh<K: ElementWiseKer<f32>>(values: &[f32]) -> TestCaseResult {
        let op = ElementWiseImpl::<K, f32>::new();
        let mut found = values.to_vec();
        op.run(&mut found).unwrap();
        let expected = values.iter().map(|x| x.tanh()).collect::<Vec<_>>();
        crate::test::check_close(&*found, &*expected)
    }
}
