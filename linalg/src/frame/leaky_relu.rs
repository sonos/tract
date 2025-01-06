#[allow(unused_macros)]
macro_rules! leaky_relu_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        ew_impl!($ti, $func, $nr, $alignment_items, $ti);
        #[cfg(test)]
        paste! {
            mod [<test_ $func>] {
                use super::*;
                leaky_relu_frame_tests!($cond, $ti, $func);
            }
        }
    };
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::{frame::element_wise::*, LADatum};
    use num_traits::{AsPrimitive, Float};
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! leaky_relu_frame_tests {
        ($cond:expr, $t: ty, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn prop(xs in proptest::collection::vec(-25f32..25.0, 0..100), alpha in 0f32..1f32) {
                    if $cond {
                        $crate::frame::leaky_relu::test::test_leaky_relu::<$ker, $t>(&*xs, alpha).unwrap()
                    }
                }
            }
            #[test]
            fn trivial() {
                if $cond {
                    $crate::frame::leaky_relu::test::test_leaky_relu::<$ker, $t>(&[-10f32], 0.0496).unwrap();
                }
            }
        };
    }

    pub fn test_leaky_relu<K: ElementWiseKer<T, T>, T: LADatum + Float>(
        values: &[f32],
        alpha: f32,
    ) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        let data = tract_data::prelude::tensor1(values);
        let data = data.cast_to::<T>().unwrap();
        let data = data.as_slice::<T>().unwrap();
        let alpha: T = tract_data::prelude::tensor0(alpha).cast_to_scalar::<T>().unwrap();
        crate::frame::element_wise::test::test_element_wise_params::<K, T, _, T>(
            data,
            |x: T| {
                if x > T::zero() {
                    x
                } else {
                    alpha * x
                }
            },
            alpha,
        )
    }
}
