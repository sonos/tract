use crate::LADatum;

use super::{ActivationKer, Op, Program, RegisterId};
use proptest::prelude::*;
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

impl Arbitrary for RegisterId {
    type Parameters = ();
    type Strategy = BoxedStrategy<RegisterId>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        proptest::prop_oneof![Just(RegisterId::A), Just(RegisterId::B), Just(RegisterId::C)].boxed()
    }
}

#[macro_export]
macro_rules! act_tests {
    ($cond:expr, $ker:ty, $ti:ty) => {
        mod acttest {
            #[allow(unused_imports)]
            use super::*;
            use num_traits::{One, Zero};
            use proptest::collection::vec;
            use proptest::prelude::*;
            use $crate::frame::activations::tests::*;
            use $crate::frame::activations::ActivationKer;
            use $crate::frame::activations::Op::*;
            use $crate::frame::activations::RegisterId;

            fn x_strat() -> impl Strategy<Value = Vec<$ti>> {
                (1usize..4).prop_flat_map(|repeat| {
                    let size = <$ker>::nr() * repeat;
                    vec(any::<$ti>(), size..size + 1)
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
                fn load_a_prop(x in x_strat(), konst in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Load(RegisterId::A, konst)], |_| konst);
                    }
                }

                #[test]
                fn load_b_prop(x in x_strat(), konst in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Load(RegisterId::B, konst)], |x| x);
                    }
                }

                #[test]
                fn load_c_prop(x in x_strat(), konst in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Load(RegisterId::C, konst)], |x| x);
                    }
                }

                #[test]
                fn move_b_to_a_prop(x in x_strat(), konst in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(
                            &x,
                            &[Load(RegisterId::B, konst), Move(RegisterId::A, RegisterId::B)],
                            |_| konst
                            );
                    }
                }

                #[test]
                fn abs_prop(x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Abs], |x| x.abs());
                    }
                }

                #[test]
                fn recip_prop(x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Recip], |x| x.recip());
                    }
                }

                #[test]
                fn add_const_prop(alpha in any::<$ti>(), x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[AddConst(alpha)], |x| x + alpha);
                    }
                }

                #[test]
                fn sub_const_prop(alpha in any::<$ti>(), x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[SubConst(alpha)], |x| x - alpha);
                    }
                }

                #[test]
                fn mul_const_prop(alpha in any::<$ti>(), x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[MulConst(alpha)], |x| x * alpha);
                    }
                }

                #[test]
                fn max_const_prop(alpha in any::<$ti>(), x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[MaxConst(alpha)], |x| x.max(alpha));
                    }
                }

                #[test]
                fn min_const_prop(alpha in any::<$ti>(), x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[MinConst(alpha)], |x| x.min(alpha));
                    }
                }

                #[test]
                fn add_prop(x in x_strat(), v in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Load(RegisterId::B, v), Add], |x| x + v);
                    }
                }

                #[test]
                fn sub_prop(x in x_strat(), v in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Load(RegisterId::B, v), Sub], |x| x - v);
                    }
                }

                #[test]
                fn mul_prop(x in x_strat(), v in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Load(RegisterId::B, v), Mul], |x| x * v);
                    }
                }

                #[test]
                fn min_prop(x in x_strat(), v in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Load(RegisterId::B, v), Min], |x| x.min(v));
                    }
                }

                #[test]
                fn max_prop(x in x_strat(), v in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x, &[Load(RegisterId::B, v), Max], |x| x.max(v));
                    }
                }


                #[test]
                fn ifposte_prop(x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x,
                            &[Load(RegisterId::B, 2 as _), Load(RegisterId::C, 3 as _), IfPosTE], 
                            |x| if x >= <$ti>::zero() { 2 as _ } else { 3 as _ });
                    }
                }

                #[test]
                fn swapbc_prop(x in x_strat()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x,
                            &[Load(RegisterId::B, 2 as _), Load(RegisterId::C, 3 as _), SwapBC, IfPosTE], 
                            |x| if x >= <$ti>::zero() { 3 as _ } else { 2 as _ });
                    }
                }

                #[test]
                fn fma_prop(x in x_strat(), b in any::<$ti>(), k in any::<$ti>()) {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(&x,
                            &[Load(RegisterId::B, b), FMA(k)],
                            |x| x * b + k);
                    }
                }
            }

                #[test]
                fn max_const_zero() {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(
                            &vec![<$ti>::zero(); <$ker>::nr()],
                            &[MaxConst(<$ti>::zero())],
                            |x| x.max(<$ti>::zero()),
                            );
                    }
                }

                #[test]
                fn max_const_big_alpha() {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(
                            &vec![<$ti>::zero(); <$ker>::nr()],
                            &[MaxConst(7.567773e37.into())],
                            |x| x.max(7.567773e37.into()),
                            );
                    }
                }

                #[test]
                fn move_b_to_a_0() {
                    if $cond {
                        run_kernel_test::<$ti, $ker>(
                            &*vec![<$ti>::zero(); <$ker>::nr()],
                            &[Load(RegisterId::B, 1.0 as _), Move(RegisterId::A, RegisterId::B)],
                            |_| 1.0 as _,
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

                    #[test]
                    fn affine_prop(x in x_strat(), alpha in any::<$ti>(), beta in any::<$ti>()) {
                        if $cond {
                            run_kernel_test::<$ti, $ker>(
                                &x,
                                &$crate::frame::activations::definitions::affine(alpha, beta).ops,
                                |x| x * alpha + beta
                                );
                        }
                    }

                    #[test]
                    fn leaky_relu_prop(x in x_strat(), alpha in any::<$ti>()) {
                        if $cond {
                            run_kernel_test::<$ti, $ker>(
                                &x,
                                &$crate::frame::activations::definitions::leaky_relu(alpha).ops,
                                |x| if x >= <$ti>::zero() { x } else { alpha * x }
                                );
                        }
                    }

                    #[test]
                    fn threshold_relu_prop(x in x_strat(), alpha in any::<$ti>()) {
                        if $cond {
                            run_kernel_test::<$ti, $ker>(
                                &x,
                                &$crate::frame::activations::definitions::threshold_relu(alpha).ops,
                                |x| if x >= alpha { x } else { <$ti>::zero() }
                                );
                        }
                    }

                    #[test]
                    fn hard_sigmoid_prop(x in x_strat(), alpha in any::<$ti>(), beta in any::<$ti>()) {
                        if $cond {
                            run_kernel_test::<$ti, $ker>(
                                &x,
                                &$crate::frame::activations::definitions::hard_sigmoid(alpha, beta).ops,
                                |x| (x * alpha + beta).min(<$ti>::one()).max(<$ti>::zero())
                                );
                        }
                    }

                    #[test]
                    fn softsign_prop(x in x_strat()) {
                        if $cond {
                            run_kernel_test::<$ti, $ker>(
                                &x,
                                &$crate::frame::activations::definitions::softsign().ops,
                                |x| x / ( <$ti>::one() + x.abs())
                                );
                        }
                    }

                    #[test]
                    fn hard_swish_prop(x in x_strat()) {
                        if $cond {
                            run_kernel_test::<$ti, $ker>(
                                &x,
                                &$crate::frame::activations::definitions::hard_swish().ops,
                                |x| (x * 1./6. + 0.5).min(<$ti>::one()).max(<$ti>::zero()) * x
                                );
                        }
                    }

                    #[test]
                    fn sigmoid_prop(x in x_strat()) {
                        if $cond {
                            run_kernel_test::<$ti, $ker>(
                                &x,
                                &$crate::frame::activations::definitions::sigmoid().ops,
                                crate::generic::sigmoid::ssigmoid
                                );
                        }
                    }
                }
            }
        };
    }
