#[allow(unused_macros)]
macro_rules! erf_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        ew_impl!($ti, $func, $nr, $alignment_items);
        #[cfg(test)]
        paste! {
            mod [<test_ $func>] {
                use super::*;
                erf_frame_tests!($cond, $ti, $func);
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
    macro_rules! erf_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-5f32..5.0, 0..100)) {
                    if $cond {
                        $crate::frame::erf::test::test_erf::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }
            #[test]
            fn trivial() {
                if $cond {
                    $crate::frame::erf::test::test_erf::<$ker, $t>(&[
                        -5f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0,
                    ])
                    .unwrap();
                }
            }
            #[test]
            fn zeros() {
                if $cond {
                    $crate::frame::erf::test::test_erf::<$ker, $t>(&[0.0; 16]).unwrap();
                }
            }
        };
    }

    pub fn test_erf<K: ElementWiseKer<T>, T: LADatum + Float>(values: &[f32]) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
        T: AsPrimitive<f32>,
    {
        let data = tract_data::prelude::tensor1(values);
        let data = data.cast_to::<T>().unwrap();
        let data = data.try_as_plain().unwrap().as_slice::<T>().unwrap();
        crate::frame::element_wise::test::test_element_wise::<K, T, _>(data, |x: T| {
            // Abramowitz & Stegun 7.1.26 six-coefficient approximation, mirroring
            // generic/erf.rs::serf so the test reference matches the production scalar path.
            const A1: f32 = 0.0705230784;
            const A2: f32 = 0.0422820123;
            const A3: f32 = 0.0092705272;
            const A4: f32 = 0.0001520143;
            const A5: f32 = 0.0002765672;
            const A6: f32 = 0.0000430638;
            let x: f32 = x.as_();
            let signum = x.signum();
            let abs = x.abs();
            let y = A6 * abs;
            let y = (A5 + y) * abs;
            let y = (A4 + y) * abs;
            let y = (A3 + y) * abs;
            let y = (A2 + y) * abs;
            let y = (A1 + y) * abs;
            let y = 1.0 - (y + 1.0).powi(16).recip();
            y.copysign(signum).as_()
        })
    }
}
