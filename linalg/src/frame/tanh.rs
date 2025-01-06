macro_rules! tanh_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        ew_impl!($ti, $func, $nr, $alignment_items);
        #[cfg(test)]
        paste! {
            mod [<test_ $func>] {
                use super::*;
                tanh_frame_tests!($cond, $ti, $func);
            }
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::element_wise::*;
    use crate::LADatum;
    use num_traits::float::Float;
    use num_traits::AsPrimitive;
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! tanh_frame_tests {
        ($cond:expr, $t:ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn tanh(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        $crate::frame::tanh::test::test_tanh::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn tanh_4_magic() {
                if $cond {
                    $crate::frame::tanh::test::test_tanh::<$ker, $t>(&[0f32, -20.0, 20.0, 0.0])
                        .unwrap()
                }
            }

            #[test]
            fn tanh_4zeros() {
                if $cond {
                    $crate::frame::tanh::test::test_tanh::<$ker, $t>(&[0.0; 4]).unwrap();
                }
            }

            #[test]
            fn tanh_20_ones() {
                if $cond {
                    $crate::frame::tanh::test::test_tanh::<$ker, $t>(&[1.0; 20]).unwrap();
                }
            }

            #[test]
            fn tanh_18_zeros() {
                if $cond {
                    $crate::frame::tanh::test::test_tanh::<$ker, $t>(&[0.0; 18]).unwrap();
                }
            }

            #[test]
            fn tanh_foo() {
                if $cond {
                    $crate::frame::tanh::test::test_tanh::<$ker, $t>(&[0.67503357]).unwrap();
                }
            }

            #[test]
            fn tanh_asymptots() {
                use tract_data::internal::*;
                use $crate::frame::element_wise::*;
                if $cond {
                    let mut input: Vec<$t> = [-100f32, 100f32]
                        .iter()
                        .map(|x| <f32 as num_traits::AsPrimitive<$t>>::as_(*x))
                        .collect();
                    let expected: Vec<$t> = [-1f32, 1f32]
                        .iter()
                        .map(|x| <f32 as num_traits::AsPrimitive<$t>>::as_(*x))
                        .collect();
                    <$ker>::ew().run(&mut input).unwrap();
                    tensor1(&input)
                        .close_enough(&tensor1(&expected), Approximation::Close)
                        .unwrap();
                }
            }
        };
    }

    pub fn test_tanh<K: ElementWiseKer<T>, T: LADatum + Float>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::setup_test_logger();
        let values: Vec<T> = values.iter().copied().map(|x| x.as_()).collect();
        crate::frame::element_wise::test::test_element_wise::<K, _, _>(&values, |x| x.tanh())
    }
}
