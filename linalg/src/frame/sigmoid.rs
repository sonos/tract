#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::{LADatum, frame::element_wise::*};
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! sigmoid_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn sigmoid(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        $crate::frame::sigmoid::test::test_sigmoid::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn sigmoid_4_magic() {
                if $cond {
                    $crate::frame::sigmoid::test::test_sigmoid::<$ker, $t>(&[
                        0f32, -20.0, 20.0, 0.0,
                    ])
                    .unwrap()
                }
            }

            #[test]
            fn sigmoid_4zeros() {
                if $cond {
                    $crate::frame::sigmoid::test::test_sigmoid::<$ker, $t>(&[0.0; 4]).unwrap();
                }
            }

            #[test]
            fn sigmoid_20_ones() {
                if $cond {
                    $crate::frame::sigmoid::test::test_sigmoid::<$ker, $t>(&[1.0; 20]).unwrap();
                }
            }

            #[test]
            fn sigmoid_18_zeros() {
                if $cond {
                    $crate::frame::sigmoid::test::test_sigmoid::<$ker, $t>(&[0.0; 18]).unwrap();
                }
            }

            #[test]
            fn sigmoid_range_on_tails() {
                if $cond {
                    $crate::frame::sigmoid::test::test_sigmoid_range::<$ker, $t>().unwrap()
                }
            }

            #[test]
            fn sigmoid_range_on_saturating_tail_sweep() {
                if $cond {
                    $crate::frame::sigmoid::test::test_sigmoid_range_exhaustive_tail::<$ker, $t>()
                        .unwrap()
                }
            }

            #[test]
            fn sigmoid_asymptots() {
                use tract_data::internal::*;
                use $crate::frame::element_wise::*;
                if $cond {
                    let mut input: Vec<$t> = [-100f32, 100f32]
                        .iter()
                        .map(|x| <f32 as num_traits::AsPrimitive<$t>>::as_(*x))
                        .collect();
                    let expected: Vec<$t> = [-0f32, 1f32]
                        .iter()
                        .map(|x| <f32 as num_traits::AsPrimitive<$t>>::as_(*x))
                        .collect();
                    <$ker>::ew().run(&mut input).unwrap();
                    // The input clamp stops short of saturation, so the tails land a few
                    // ulps inside [0, 1] instead of on it. `Close` still fits f16, whose
                    // own clamp costs it about 1e-3; f32's needs the atol widened, and a
                    // relative bound is no use against the 0 asymptote.
                    let approx = if <$t>::datum_type() == f16::datum_type() {
                        Approximation::Close
                    } else {
                        Approximation::Custom(1e-6, 1e-6, 0.)
                    };
                    tensor1(&input).close_enough(&tensor1(&expected), approx).unwrap();
                }
            }
        };
    }

    /// Assert every output of a sigmoid kernel lands in `[0, 1]`, the range its consumers
    /// rely on and which the tail cancellation of a `p / q + 0.5` kernel can step outside.
    pub fn test_sigmoid_range<K: ElementWiseKer<T>, T: LADatum + Float>() -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::frame::element_wise::test::test_element_wise_invariant::<K, T>(
            "a result in [0, 1]",
            |_, y| y >= T::zero() && y <= T::one(),
        )
    }

    /// Assert the same range over every `f32` of the saturating tail, `[13, 18]` and its
    /// negation.
    ///
    /// A kernel that carries no output clamp holds its range only because its input clamp
    /// stops short of where the `+ 0.5` cancellation runs under the rounding error of
    /// `p / q`. The inputs that cross sit a few `1e-7` apart, so the grid
    /// [`test_sigmoid_range`] sweeps steps over them: only enumerating the tail pins the
    /// clamp down. `f16` is already enumerated whole, and skips this.
    pub fn test_sigmoid_range_exhaustive_tail<K: ElementWiseKer<T>, T: LADatum + Float>()
    -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        if T::datum_type() != <f32 as tract_data::prelude::Datum>::datum_type() {
            return Ok(());
        }
        crate::setup_test_logger();
        const CHUNK: usize = 1 << 16;
        let end = 18f32.to_bits();
        for sign in [1f32, -1f32] {
            let mut inputs: Vec<T> = Vec::with_capacity(CHUNK);
            let mut outputs: Vec<T> = Vec::with_capacity(CHUNK);
            let mut bits = 13f32.to_bits();
            while bits <= end {
                inputs.clear();
                while bits <= end && inputs.len() < CHUNK {
                    inputs.push((sign * f32::from_bits(bits)).as_());
                    bits += 1;
                }
                outputs.clear();
                outputs.extend_from_slice(&inputs);
                K::ew().run(&mut outputs).unwrap();
                for (x, y) in inputs.iter().zip(outputs.iter()) {
                    proptest::prop_assert!(
                        *y >= T::zero() && *y <= T::one(),
                        "{}({x:?}) returned {y:?}, expected a result in [0, 1]",
                        K::name()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn test_sigmoid<K: ElementWiseKer<T>, T: LADatum + Float>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::setup_test_logger();
        let values: Vec<T> = values.iter().copied().map(|x| x.as_()).collect();
        crate::frame::element_wise::test::test_element_wise::<K, _, _>(&values, |x| {
            (1f32).as_() / (1f32.as_() + (-x).exp())
        })
    }
}
