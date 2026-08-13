#[allow(unused_macros)]
macro_rules! gelu_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        ew_impl!($ti, $func, $nr, $alignment_items);
        #[cfg(test)]
        paste! {
            mod [<test_ $func>] {
                use super::*;
                gelu_frame_tests!($cond, $ti, $func);
            }
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::LADatum;
    use crate::frame::element_wise::*;
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! gelu_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-10f32..10.0, 0..100)) {
                    if $cond {
                        $crate::frame::gelu::test::test_gelu::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }
            #[test]
            fn trivial() {
                if $cond {
                    $crate::frame::gelu::test::test_gelu::<$ker, $t>(&[-5f32, -1.0, 0.0, 1.0, 5.0])
                        .unwrap();
                }
            }

            #[test]
            fn sign_on_tails() {
                if $cond {
                    $crate::frame::gelu::test::test_gelu_sign::<$ker, $t>().unwrap()
                }
            }

            #[test]
            fn magnitude_on_tails() {
                if $cond {
                    $crate::frame::gelu::test::test_gelu_magnitude::<$ker, $t>().unwrap()
                }
            }
        };
    }

    /// Assert every output of a GELU kernel carries the sign of its input: `gelu(x) =
    /// 0.5 * x * (1 + tanh(..))` and the tanh-form factor is positive, so an internal tanh
    /// that dips below -1 makes the factor negative and flips the sign of the result.
    pub fn test_gelu_sign<K: ElementWiseKer<T>, T: LADatum + Float>() -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::frame::element_wise::test::test_element_wise_invariant::<K, T>(
            "the sign of the input",
            |x, y| if x < T::zero() { y <= T::zero() } else { y >= T::zero() },
        )
    }

    /// Assert no output of a GELU kernel exceeds its input in magnitude. The tanh-form
    /// factor is below 1, so `gelu` contracts towards zero — negative inputs included,
    /// where the output is negative but never more negative than the input. An internal
    /// tanh above 1 pushes the factor past 1 and the result past its input.
    pub fn test_gelu_magnitude<K: ElementWiseKer<T>, T: LADatum + Float>() -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::frame::element_wise::test::test_element_wise_invariant::<K, T>(
            "a magnitude not above the input's",
            |x, y| y.abs() <= x.abs(),
        )
    }

    pub fn test_gelu<K: ElementWiseKer<T>, T: LADatum + Float>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        let data = tract_data::prelude::tensor1(values);
        let data = data.cast_to::<T>().unwrap();
        let data = data.try_as_plain().unwrap().as_slice::<T>().unwrap();
        // Tanh-form GELU (pow=3): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        crate::frame::element_wise::test::test_element_wise::<K, T, _>(data, |x: T| {
            let half: T = 0.5f32.as_();
            let one: T = 1f32.as_();
            let coef: T = 0.044715f32.as_();
            let sqrt_2_over_pi: T = 0.7978845608028654f32.as_();
            let inner = sqrt_2_over_pi * (x + coef * x * x * x);
            half * x * (one + inner.tanh())
        })
    }
}
