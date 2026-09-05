#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::LADatum;
    use crate::frame::element_wise::*;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;

    #[macro_export]
    macro_rules! exp_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-20f32..20.0, 0..100)) {
                    if $cond {
                        $crate::frame::exp::test::test_exp::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }
            #[test]
            fn trivial() {
                if $cond {
                    $crate::frame::exp::test::test_exp::<$ker, $t>(&[
                        0f32, 1.0, -1.0, 0.5, -0.5, 10.0, -10.0, 80.0, -80.0,
                    ])
                    .unwrap();
                }
            }
            #[test]
            fn zeros() {
                if $cond {
                    $crate::frame::exp::test::test_exp::<$ker, $t>(&[0.0; 16]).unwrap();
                }
            }
            #[test]
            fn whole_range() {
                if $cond {
                    $crate::frame::exp::test::test_exp_ulp::<$ker, $t>().unwrap();
                }
            }
            #[test]
            fn specials() {
                if $cond {
                    $crate::frame::exp::test::test_exp_specials::<$ker, $t>().unwrap();
                }
            }
        };
    }

    pub fn test_exp<K: ElementWiseKer<T>, T>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
        T: LADatum + Float + AsPrimitive<f32>,
    {
        let data = tract_data::prelude::tensor1(values);
        let data = data.cast_to::<T>().unwrap();
        let data = data.try_as_plain().unwrap().as_slice::<T>().unwrap();
        crate::frame::element_wise::test::test_element_wise::<K, T, _>(data, |x: T| {
            let x: f32 = x.as_();
            ((x as f64).exp() as f32).as_()
        })
    }

    /// A fine grid over the whole range that answers with a finite non-zero number,
    /// against a correctly rounded `exp`: the reduction's error grows with `|x|`, and the
    /// tails are where the two halves of the scale and the clamp are decided.
    pub fn test_exp_ulp<K: ElementWiseKer<T>, T>() -> TestCaseResult
    where
        T: LADatum + Float + AsPrimitive<f32>,
        f32: AsPrimitive<T>,
    {
        if T::datum_type() != f32::datum_type() {
            return Ok(());
        }
        crate::setup_test_logger();
        let input: Vec<T> =
            (-104_000..89_000).map(|i| (i as f32 / 1000.0).as_()).step_by(3).collect();
        let expected: Vec<T> = input
            .iter()
            .map(|x| {
                let x: f32 = x.as_();
                ((x as f64).exp() as f32).as_()
            })
            .collect();
        let mut found = input;
        K::ew().run(&mut found).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected), Approximation::Ulp(3))
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))
    }

    /// What the clamp answers with off its own ends, and where the exact answers are
    /// exact.
    pub fn test_exp_specials<K: ElementWiseKer<T>, T>() -> TestCaseResult
    where
        T: LADatum + Float + AsPrimitive<f32>,
        f32: AsPrimitive<T>,
    {
        crate::setup_test_logger();
        let input: Vec<T> = [0f32, -0.0, 200.0, -200.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]
            .iter()
            .map(|x| x.as_())
            .collect();
        let mut found = input.clone();
        K::ew().run(&mut found).unwrap();
        for (x, y) in input.iter().zip(found.iter()) {
            let x: f32 = x.as_();
            let y: f32 = y.as_();
            let ok = if x.is_nan() {
                y.is_nan()
            } else if x == 0.0 {
                y == 1.0
            } else if x > 100.0 {
                y == f32::INFINITY
            } else {
                y == 0.0
            };
            proptest::prop_assert!(ok, "{}({x}) returned {y}", K::name());
        }
        Ok(())
    }
}
