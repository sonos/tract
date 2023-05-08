use crate::LADatum;

use super::{ActivationKer, Op, Program};
use Op::*;

pub fn noop<T: LADatum>() -> Program<T> {
    Program { ops: vec![] }
}

pub fn max_const<T: LADatum>(c: T) -> Program<T> {
    Program { ops: vec![MaxConst(c)] }
}

pub fn run_kernel_test<TI: LADatum, K: ActivationKer<TI>>(
    input: &[TI],
    prog: &[Op<TI>],
    refer: impl Fn(TI) -> TI,
) {
    let mut tensor =
        tract_data::prelude::Tensor::zero_aligned::<TI>(&[input.len()], K::alignment_bytes())
            .unwrap();
    tensor.as_slice_mut::<TI>().unwrap().copy_from_slice(input);
    let expected: Vec<TI> = input.iter().cloned().map(|x| refer(x)).collect();
    let expected = tract_data::prelude::tensor1(&expected);
    let prog = Program { ops: prog.to_vec() };
    let prog = prog.translate();
    K::run(&prog.ops, tensor.as_slice_mut::<TI>().unwrap());
    expected.close_enough(&tensor, true).unwrap();
}

#[macro_export]
macro_rules! act_tests {
    ($cond:expr, $ker:ty, $ti:ty) => {
        mod acttest {
            #[allow(unused_imports)]
            use super::*;
            use $crate::frame::activations::ActivationKer;
            use $crate::frame::activations::tests::*;
            use $crate::frame::activations::Op::*;
            use num_traits::Zero;
            use proptest::prelude::*;
            use proptest::collection::vec;

            fn x_strat() -> impl Strategy<Value = Vec<$ti>> {
                (1usize..4).prop_flat_map(|repeat| {
                    let size = <$ker>::nr() * repeat;
                    vec(any::<$ti>(), size..size+1)
                })
            }

            proptest::proptest! {
                #[test]
                fn noop(x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[], |x| x);
                    }
                }

                #[test]
                fn max_const_prop(alpha in any::<$ti>(), x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[MaxConst(alpha)], |x| x.max(alpha));
                    }
                }
            }

            #[test]
            fn max_const_zero() {
                if $cond {
                    run_kernel_test::<$ti, $ker>(
                        &vec!(<$ti>::zero(); <$ker>::nr()),
                        &[MaxConst(<$ti>::zero())],
                        |x| x.max(<$ti>::zero())
                    );
                }
            }

            #[test]
            fn max_const_big_alpha() {
                if $cond {
                    run_kernel_test::<$ti, $ker>(
                        &vec!(<$ti>::zero(); <$ker>::nr()),
                        &[MaxConst(7.567773e37.into())],
                        |x| x.max(7.567773e37.into())
                    );
                }
            }

            proptest::proptest! {
                #[test]
                fn relu_prop(x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(
                            &x,
                            &$crate::frame::activations::definitions::relu().ops,
                            |x| x.max(<$ti>::zero())
                        );
                    }
                }
            }
            /*
            prop_act_e2e!($cond, $ti, $ker, affine(alpha, beta));
            prop_act_e2e!($cond, $ti, $ker, leaky_relu(alpha));
            prop_act_e2e!($cond, $ti, $ker, threshold_relu(alpha));
            prop_act_e2e!($cond, $ti, $ker, softsign());
            prop_act_e2e!($cond, $ti, $ker, hardswish());
            /*
               prop_activation!($cond, $ti, $ker, sigmoid());
               prop_activation!($cond, $ti, $ker, exp2f());
               */
            */
        }
    };
}
