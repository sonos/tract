use crate::frame::mmm::fuse::RoundingPolicy;
use crate::frame::mmm::MatMatMulKer;
use crate::generic::rounding::ScaleShiftAndRound;
use crate::mmm::{FusedKerSpec, FusedSpec};
use crate::Scaler;
use proptest::prelude::*;

use super::fuse::fused_ops;

#[derive(Debug, new)]
pub struct QScaleProblem<K>
where
    K: MatMatMulKer<Acc = i32>,
{
    pub ker: K,
    pub c: Vec<i32>,
    pub scaler: Scaler,
    pub boo: std::marker::PhantomData<K>,
}

impl<K> Arbitrary for QScaleProblem<K>
where
    K: MatMatMulKer<Acc = i32> + 'static + Copy + Default,
{
    type Parameters = K;
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(ker: K) -> Self::Strategy {
        use RoundingPolicy::*;
        let len = ker.mr() * ker.nr();
        (
            proptest::collection::vec(-20i32..20, len..=len),
            -5i32..5,
            prop_oneof!(Just(1f32), 0f32..1f32),
            proptest::prop_oneof![
                Just(Zero),
                Just(Away),
                Just(PlusInf),
                Just(MinusInf),
                Just(Odd),
                Just(Even)
            ],
        )
            .prop_map(|(c, scale_pot, scale_mult, policy)| QScaleProblem {
                ker: K::default(),
                c,
                scaler: Scaler::new(scale_mult * 2f32.powi(scale_pot), policy),
                boo: std::marker::PhantomData,
            })
            .boxed()
    }
}

impl<K> QScaleProblem<K>
where
    K: MatMatMulKer<Acc = i32>,
{
    pub fn run(&self) {
        if let FusedSpec::QScale(shift, policy, mult) = self.scaler.as_fused_spec() {
            fused_ops::<K, i32, _>(
                self.ker,
                &self.c,
                &[FusedKerSpec::QScale(shift, policy, mult)],
                |_, _, c| c.q_scale(self.scaler),
            )
        } else if let FusedSpec::RoundingShiftRight(shift, policy) = self.scaler.as_fused_spec() {
            fused_ops::<K, i32, _>(
                self.ker,
                &self.c,
                &[FusedKerSpec::RoundingShiftRight(shift, policy)],
                |_, _, c| c.q_shr(shift, policy),
            )
        } else if let FusedSpec::ShiftLeft(shift) = self.scaler.as_fused_spec() {
            fused_ops::<K, i32, _>(
                self.ker,
                &self.c,
                &[FusedKerSpec::ShiftLeft(shift)],
                |_, _, c| c.q_shl(shift),
            )
        } else {
            unreachable!()
        }
    }
}

pub fn return_c_scale_bigpot<K>(ker: K)
where
    K: MatMatMulKer<Acc = i32>,
{
    let len = ker.mr() * ker.nr();
    let v: Vec<i32> = (-(len as i32) / 2..).take(len).collect();
    fused_ops::<K, i32, _>(ker, &v, &[FusedKerSpec::ShiftLeft(1)], |_, _, c| c.q_shl(1))
}

#[macro_export]
macro_rules! mmm_q_scale_tests {
    ($cond:expr, $ker:ident) => {
        use $crate::frame::mmm::fuse::RoundingPolicy;
        use $crate::frame::mmm::tests::q_scale::QScaleProblem;
        use $crate::frame::mmm::MatMatMulKer;
        use $crate::generic::Scaler;
        use proptest::prelude::*;

        // FIXME: Scaler should be arbitrary
        macro_rules! test_q_scale {
            ($policy: ident) => {
                paste! {
                    #[test]
                    fn [<return_q_scale_halfpos_ $policy:lower>]() {
                        if $cond {
                            let len = ($ker.mr() * $ker.nr()) as i64;
                            let v = (0..len).map(|i| (i - len / 2) as i32).collect();
                            QScaleProblem::new($ker, v, Scaler::new(0.5f32, RoundingPolicy::$policy)).run()
                        }
                    }

                    #[test]
                    fn [<return_q_scale_halfneg_ $policy:lower>]() {
                        if $cond {
                            let len = ($ker.mr() * $ker.nr()) as i64;
                            let v = (0..len).map(|i| (i - len / 2) as i32).collect();
                            QScaleProblem::new($ker, v, Scaler::new(-0.5f32, RoundingPolicy::$policy)).run()
                        }
                    }

                    #[test]
                    fn [<return_q_scale_pot_ $policy:lower>]() {
                        if $cond {
                            let len = ($ker.mr() * $ker.nr()) as i64;
                            let v = (0..len).map(|i| (i - len / 2) as i32).collect();
                            QScaleProblem::new($ker, v, Scaler::new(0.25f32, RoundingPolicy::$policy)).run()
                        }
                    }

                    #[test]
                    fn [<return_q_scale_nonpot_ $policy:lower>]() {
                        if $cond {
                            let len = ($ker.mr() * $ker.nr()) as i64;
                            let v = (0..len).map(|i| (i - len / 2) as i32).collect();
                            QScaleProblem::new($ker, v, Scaler::new(1f32 / 5., RoundingPolicy::$policy)).run()
                        }
                    }

                    #[test]
                    fn [<return_q_scale_bigpot_ $policy:lower>]() {
                        if $cond {
                            let len = ($ker.mr() * $ker.nr()) as i64;
                            let v = (0..len).map(|i| (i - len / 2) as i32).collect();
                            QScaleProblem::new($ker, v, Scaler::new(4f32, RoundingPolicy::$policy)).run()
                        }
                    }

                    #[test]
                    fn [<return_q_scale_bignonpot_ $policy:lower>]() {
                        if $cond {
                            let len = ($ker.mr() * $ker.nr()) as i64;
                            let v = (0..len).map(|i| (i - len / 2) as i32).collect();
                            QScaleProblem::new($ker, v, Scaler::new(14., RoundingPolicy::$policy)).run()
                        }
                    }
                }
            }
        }

        test_q_scale!(Zero);
        test_q_scale!(Away);
        test_q_scale!(MinusInf);
        test_q_scale!(PlusInf);
        test_q_scale!(Even);
        test_q_scale!(Odd);

        proptest::proptest! {
            #[test]
            fn return_q_scale_prop(pb in any_with::<QScaleProblem::<_>>($ker)) {
                if $cond {
                    pb.run()
                }
            }
        }

        #[test]
        fn return_c_scale_bigpot() {
            if $cond {
                crate::frame::mmm::tests::q_scale::return_c_scale_bigpot::<_>($ker)
            }
        }
    };
}
