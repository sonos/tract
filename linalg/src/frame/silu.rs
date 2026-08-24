#[allow(unused_macros)]
macro_rules! silu_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        ew_impl!($ti, $func, $nr, $alignment_items);
        #[cfg(test)]
        paste! {
            mod [<test_ $func>] {
                use super::*;
                silu_frame_tests!($cond, $ti, $func);
            }
        }
    };
    ($arch:ident; $ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        ew_impl!($arch; $ti, $func, $nr, $alignment_items);
        #[cfg(test)]
        paste! {
            mod [<test_ $func>] {
                use super::*;
                silu_frame_tests!($cond, $ti, $func);
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
    macro_rules! silu_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-10f32..10.0, 0..100)) {
                    if $cond {
                        $crate::frame::silu::test::test_silu::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }
            #[test]
            fn trivial() {
                if $cond {
                    $crate::frame::silu::test::test_silu::<$ker, $t>(&[-5f32, -1.0, 0.0, 1.0, 5.0])
                        .unwrap();
                }
            }
            #[test]
            fn sign_on_tails() {
                if $cond {
                    $crate::frame::silu::test::test_silu_sign::<$ker, $t>().unwrap()
                }
            }

            #[test]
            fn sign_on_saturating_tail_sweep() {
                if $cond {
                    $crate::frame::silu::test::test_silu_sign_exhaustive_tail::<$ker, $t>().unwrap()
                }
            }

            #[test]
            fn tails() {
                if $cond {
                    $crate::frame::silu::test::test_silu::<$ker, $t>(&[
                        -100.0, -30.0, -14.5, -12.0, 12.0, 14.5, 30.0, 100.0,
                    ])
                    .unwrap();
                }
            }
        };
    }

    /// Assert every output of a SiLU kernel carries the sign of its input: `SiLU(x) =
    /// x * sigmoid(x)` and sigmoid is positive, so a sigmoid factor that dips below zero
    /// flips the sign of the whole result.
    pub fn test_silu_sign<K: ElementWiseKer<T>, T: LADatum + Float>() -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::frame::element_wise::test::test_element_wise_invariant::<K, T>(
            "the sign of the input",
            |x, y| if x < T::zero() { y <= T::zero() } else { y >= T::zero() },
        )
    }

    /// Assert the same sign over every `f32` of the saturating tail, `[13, 18]` and its
    /// negation.
    ///
    /// A kernel that carries no floor on its sigmoid factor holds the sign only because
    /// the argument clamp stops short of where that factor's `+ 0.5` runs under the
    /// rounding error of `p / q`. The inputs that cross sit a few `1e-7` apart, so the
    /// grid [`test_silu_sign`] sweeps steps over them: only enumerating the tail pins the
    /// clamp down. `f16` is already enumerated whole, and skips this.
    pub fn test_silu_sign_exhaustive_tail<K: ElementWiseKer<T>, T: LADatum + Float>()
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
                    let signed = if *x < T::zero() { *y <= T::zero() } else { *y >= T::zero() };
                    proptest::prop_assert!(
                        signed,
                        "{}({x:?}) returned {y:?}, expected the sign of the input",
                        K::name()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn test_silu<K: ElementWiseKer<T>, T: LADatum + Float>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        let data = tract_data::prelude::tensor1(values);
        let data = data.cast_to::<T>().unwrap();
        let data = data.try_as_plain().unwrap().as_slice::<T>().unwrap();
        crate::frame::element_wise::test::test_element_wise::<K, T, _>(data, |x: T| {
            let one: T = 1f32.as_();
            let neg_x = T::zero() - x;
            let sigmoid = one / (one + neg_x.exp());
            x * sigmoid
        })
    }
}
