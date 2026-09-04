#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::LADatum;
    use crate::frame::element_wise::*;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;

    #[macro_export]
    macro_rules! ln_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(1e-6f32..1e6, 0..100)) {
                    if $cond {
                        $crate::frame::ln::test::test_ln::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }
            #[test]
            fn trivial() {
                if $cond {
                    $crate::frame::ln::test::test_ln::<$ker, $t>(&[
                        1f32,
                        0.5,
                        2.0,
                        std::f32::consts::E,
                        0.1,
                        10.0,
                        1e-30,
                        1e30,
                    ])
                    .unwrap();
                }
            }
            #[test]
            fn ones() {
                if $cond {
                    $crate::frame::ln::test::test_ln::<$ker, $t>(&[1.0; 16]).unwrap();
                }
            }
            #[test]
            fn every_binade() {
                if $cond {
                    $crate::frame::ln::test::test_ln_ulp::<$ker, $t>().unwrap();
                }
            }
            #[test]
            fn specials() {
                if $cond {
                    $crate::frame::ln::test::test_ln_specials::<$ker, $t>().unwrap();
                }
            }
        };
    }

    pub fn test_ln<K: ElementWiseKer<T>, T>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
        T: LADatum + Float + AsPrimitive<f32>,
    {
        let data = tract_data::prelude::tensor1(values);
        let data = data.cast_to::<T>().unwrap();
        let data = data.try_as_plain().unwrap().as_slice::<T>().unwrap();
        crate::frame::element_wise::test::test_element_wise::<K, T, _>(data, |x: T| {
            let x: f32 = x.as_();
            ((x as f64).ln() as f32).as_()
        })
    }

    /// Every 65537th f32 bit pattern from the smallest subnormal up, against a correctly
    /// rounded `ln`: a grid over the values would miss the subnormals and the binades
    /// either side of one, which is where the mantissa split and the subnormal prescale
    /// are decided.
    pub fn test_ln_ulp<K: ElementWiseKer<T>, T>() -> TestCaseResult
    where
        T: LADatum + Float + AsPrimitive<f32>,
        f32: AsPrimitive<T>,
    {
        if T::datum_type() != f32::datum_type() {
            return Ok(());
        }
        crate::setup_test_logger();
        let input: Vec<T> =
            (1..0x7f7f_ffffu32).step_by(65537).map(|bits| f32::from_bits(bits).as_()).collect();
        let expected: Vec<T> = input
            .iter()
            .map(|x| {
                let x: f32 = x.as_();
                ((x as f64).ln() as f32).as_()
            })
            .collect();
        let mut found = input;
        K::ew().run(&mut found).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected), Approximation::Ulp(2))
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))
    }

    /// The IEEE answers off the fit's domain, which no tolerance-based comparison can
    /// state: they are exact values or NaN.
    pub fn test_ln_specials<K: ElementWiseKer<T>, T>() -> TestCaseResult
    where
        T: LADatum + Float + AsPrimitive<f32>,
        f32: AsPrimitive<T>,
    {
        crate::setup_test_logger();
        let input: Vec<T> = [0f32, -0.0, -1.0, f32::NEG_INFINITY, f32::INFINITY, f32::NAN, 1.0]
            .iter()
            .map(|x| x.as_())
            .collect();
        let mut found = input.clone();
        K::ew().run(&mut found).unwrap();
        for (x, y) in input.iter().zip(found.iter()) {
            let x: f32 = x.as_();
            let y: f32 = y.as_();
            let ok = if x < 0.0 || x.is_nan() {
                y.is_nan()
            } else if x == 0.0 {
                y == f32::NEG_INFINITY
            } else if x.is_infinite() {
                y == f32::INFINITY
            } else {
                y == 0.0
            };
            proptest::prop_assert!(ok, "{}({x}) returned {y}", K::name());
        }
        Ok(())
    }
}
