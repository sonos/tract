use crate::LADatum;

use super::{Op, Program};
use Op::*;

pub fn noop<T: LADatum>() -> Program<T> {
    Program { ops: vec![] }
}

pub fn max_const<T: LADatum>(c: T) -> Program<T> {
    Program { ops: vec![MaxConst(c)] }
}

macro_rules! prop_act_e2e {
    ($cond:expr, $ti: ty, $ker: ty, $name: ident ( $($param:ident),* )) => {
        proptest::proptest! {
            #[test]
            fn $name(
                x in proptest::prelude::any::<$ti>(),
                repeat in 1usize..4,
                $($param in proptest::prelude::any::<$ti>()),*)
            {
                use crate::frame::activations::ActivationKer;
                if $cond {
                    let mut input = tract_data::prelude::Tensor::zero_aligned::<$ti>(&[<$ker>::nr() * repeat], <$ker>::alignment_bytes()).unwrap();
                    input.fill_t::<$ti>(x).unwrap();
                    let prog = crate::frame::activations::definitions::$name($($param),*).translate();
                    <$ker>::run(&prog.ops, input.as_slice_mut::<$ti>().unwrap());
                    let expected = crate::frame::activations::reference::$name(x, $($param),*);
                    let mut output = tract_data::prelude::Tensor::zero_aligned::<$ti>(&[<$ker>::nr() * repeat], <$ker>::alignment_bytes()).unwrap();
                    output.fill_t::<$ti>(expected).unwrap();
                    output.close_enough(&input, true).unwrap();
                }
            }
        }
    }
}

macro_rules! prop_act_unit {
    ($cond:expr, $ti: ty, $ker: ty, $name: ident ( $($param:ident),* ), $refer: expr) => {
        proptest::proptest! {
            #[test]
            fn $name(
                x in proptest::prelude::any::<$ti>(),
                repeat in 1usize..4,
                $($param in proptest::prelude::any::<$ti>()),*)
            {
                use crate::frame::activations::ActivationKer;
                if $cond {
                    let mut input = tract_data::prelude::Tensor::zero_aligned::<$ti>(&[<$ker>::nr() * repeat], <$ker>::alignment_bytes()).unwrap();
                    input.fill_t::<$ti>(x).unwrap();
                    let expected:Vec<$ti> = input.as_slice::<$ti>().unwrap().iter().cloned().map(|x| $refer(x, $($param),*)).collect();
                    let prog = crate::frame::activations::tests::$name($($param),*).translate();
                    <$ker>::run(&prog.ops, input.as_slice_mut::<$ti>().unwrap());

                    let expected = tract_data::prelude::tensor1(&expected);
                    expected.close_enough(&input, true).unwrap();
                }
            }
        }
    }
}

#[macro_export]
macro_rules! act_tests {
    ($cond:expr, $ker:ty, $ti:ty) => {
        prop_act_unit!($cond, $ti, $ker, noop(), |x| x);
        prop_act_unit!($cond, $ti, $ker, max_const(alpha), |x: $ti, alpha| x.max(alpha));

        #[test]
        fn max_const_0() {
            use crate::frame::activations::ActivationKer;
            if $cond {
                let mut input = tract_data::prelude::Tensor::zero_aligned::<$ti>(
                    &[<$ker>::nr()],
                    <$ker>::alignment_bytes(),
                )
                .unwrap();
                input.fill_t::<$ti>(0.0).unwrap();
                let expected: Vec<$ti> =
                    input.as_slice::<$ti>().unwrap().iter().cloned().map(|x| x.max(0f32)).collect();
                let prog = crate::frame::activations::tests::max_const(0f32).translate();
                <$ker>::run(&prog.ops, input.as_slice_mut::<$ti>().unwrap());

                let expected = tract_data::prelude::tensor1(&expected);
                expected.close_enough(&input, true).unwrap();
            }
        }

        #[test]
        fn max_const_big_alpha() {
            use crate::frame::activations::ActivationKer;
            if $cond {
                let mut input = tract_data::prelude::Tensor::zero_aligned::<$ti>(
                    &[<$ker>::nr()],
                    <$ker>::alignment_bytes(),
                )
                .unwrap();
                input.fill_t::<$ti>(0.0).unwrap();
                let expected: Vec<$ti> = input
                    .as_slice::<$ti>()
                    .unwrap()
                    .iter()
                    .cloned()
                    .map(|x| x.max(7.567773e37))
                    .collect();
                let prog = crate::frame::activations::tests::max_const(7.567773e37).translate();
                <$ker>::run(&prog.ops, input.as_slice_mut::<$ti>().unwrap());

                let expected = tract_data::prelude::tensor1(&expected);
                expected.close_enough(&input, true).unwrap();
            }
        }

        prop_act_e2e!($cond, $ti, $ker, relu());
        prop_act_e2e!($cond, $ti, $ker, affine(alpha, beta));
        prop_act_e2e!($cond, $ti, $ker, leaky_relu(alpha));
        prop_act_e2e!($cond, $ti, $ker, threshold_relu(alpha));
        prop_act_e2e!($cond, $ti, $ker, softsign());
        prop_act_e2e!($cond, $ti, $ker, hardswish());
        /*
        prop_activation!($cond, $ti, $ker, sigmoid());
        prop_activation!($cond, $ti, $ker, exp2f());
        */
    };
}
