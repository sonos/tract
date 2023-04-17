use crate::LADatum;

use super::{Program, Op};
use Op::*;

pub fn noop<T: LADatum>() -> Program<T> {
    Program { ops: vec![Done], csts: vec![] }
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
                    let prog = crate::frame::activations::definitions::$name($($param),*);
                    <$ker>::run(&prog.ops, &prog.csts, &mut input.as_slice_mut::<$ti>().unwrap());
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
                    let refer2: fn($ti) -> $ti = $refer;
                    let expected:Vec<$ti> = input.as_slice::<$ti>().unwrap().iter().cloned().map(refer2).collect();
                    let prog = crate::frame::activations::tests::$name($($param),*);
                    <$ker>::run(&prog.ops, &prog.csts, &mut input.as_slice_mut::<$ti>().unwrap());

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

