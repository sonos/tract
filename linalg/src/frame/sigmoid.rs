#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::element_wise::*;
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! sigmoid_frame_tests {
        ($cond:expr, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn sigmoid(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        crate::frame::sigmoid::test::test_sigmoid::<$ker>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn sigmoid_4_magic() {
                if $cond {
                    crate::frame::sigmoid::test::test_sigmoid::<$ker>(&[0f32, -20.0, 20.0, 0.0])
                        .unwrap()
                }
            }

            #[test]
            fn sigmoid_4zeros() {
                if $cond {
                    crate::frame::sigmoid::test::test_sigmoid::<$ker>(&[0.0; 4]).unwrap();
                }
            }

            #[test]
            fn sigmoid_20_ones() {
                if $cond {
                    crate::frame::sigmoid::test::test_sigmoid::<$ker>(&[1.0; 20]).unwrap();
                }
            }

            #[test]
            fn sigmoid_18_zeros() {
                if $cond {
                    crate::frame::sigmoid::test::test_sigmoid::<$ker>(&[0.0; 18]).unwrap();
                }
            }
        };
    }

    pub fn test_sigmoid<K: ElementWiseKer<f32>>(values: &[f32]) -> TestCaseResult {
        crate::frame::element_wise::test::test_element_wise::<K, _, _>(values, |x| {
            1.0 / (1.0 + (-x).exp())
        })
    }
}
