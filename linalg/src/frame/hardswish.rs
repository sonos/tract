#[allow(unused_macros)]
macro_rules! hardswish_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        ew_impl!($ti, $func, $nr, $alignment_items);
        #[cfg(test)]
        paste! {
            mod [<test_ $func>] {
                use super::*;
                hardswish_frame_tests!($cond, $ti, $func);
            }
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::LADatum;
    use crate::frame::element_wise::*;
    use num_traits::{AsPrimitive, Float, Zero};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! hardswish_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        $crate::frame::hardswish::test::test_hardswish::<$ker, $t>(&*xs).unwrap()
                    }
                }
            }
            #[test]
            fn trivial() {
                if $cond {
                    $crate::frame::hardswish::test::test_hardswish::<$ker, $t>(&[
                        -10f32, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0,
                    ])
                    .unwrap();
                }
            }
        };
    }

    pub fn test_hardswish<K: ElementWiseKer<T>, T: LADatum + Float>(
        values: &[f32],
    ) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        let data = tract_data::prelude::tensor1(values);
        let data = data.cast_to::<T>().unwrap();
        let data = data.try_as_plain().unwrap().as_slice::<T>().unwrap();
        crate::frame::element_wise::test::test_element_wise::<K, T, _>(data, |x: T| {
            let three: T = 3f32.as_();
            let six: T = 6f32.as_();
            let zero: T = T::zero();
            let inv6: T = (1f32 / 6f32).as_();
            let relu6 = ((x + three).min(six)).max(zero);
            x * relu6 * inv6
        })
    }
}
