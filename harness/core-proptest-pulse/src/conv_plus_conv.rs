use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use tract_hir::internal::*;
use tract_hir::ops::cnn::*;
use tract_hir::prelude::tract_itertools::Itertools;

use super::*;

#[derive(Debug, Clone)]
struct ConvOp {
    stride: usize,
    dilation: usize,
    ker: Tensor,
    padding: PaddingSpec,
}

impl ConvOp {
    fn chain(&self, name: &str, model: &mut InferenceModel, after: OutletId) -> OutletId {
        let filters = model.add_const(format!("{}-kernel", name), self.ker.clone()).unwrap();
        let mut conv = tract_hir::ops::cnn::Conv::default();
        conv.dilations = Some(tvec!(self.dilation));
        conv.strides = Some(tvec!(self.stride));
        conv.padding = self.padding.clone();
        model.wire_node(name, expand(conv), &[after, filters]).unwrap()[0]
    }
}

impl Arbitrary for ConvOp {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..3, 1usize..3, 1usize..4)
            .prop_flat_map(|(stride, dil, ker)| {
                let padding = (ker - 1) * dil;
                let explicit = (0..=padding).prop_map(move |right| {
                    PaddingSpec::Explicit(tvec!(padding - right), tvec!(right), false)
                });
                (Just((stride, dil, ker)), prop_oneof![Just(PaddingSpec::Valid), explicit])
            })
            .prop_map(|((stride, dilation, ker), padding)| ConvOp {
                stride,
                dilation,
                ker: t(ker),
                padding,
            })
            .boxed()
    }
}

#[derive(Debug, Clone)]
struct ConvPlusConvProblem {
    input: Tensor,
    pulse: usize,
    convs: Vec<ConvOp>,
}

impl Arbitrary for ConvPlusConvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (proptest::collection::vec(ConvOp::arbitrary(), 1..4), 1usize..4)
            .prop_flat_map(|(convs, pulse_factor)| {
                let pulse = convs.iter().map(|cv| cv.stride).product::<usize>() * pulse_factor;
                let min_input = Self::min_input_size(&convs);
                (Just(convs), Just(pulse), min_input..3 * min_input)
            })
            .prop_map(|(convs, pulse, input)| ConvPlusConvProblem { input: t(input), pulse, convs })
            .boxed()
    }
}

impl ConvPlusConvProblem {
    pub fn min_input_size(ops: &[ConvOp]) -> usize {
        let model = Self::model(ops);
        let dims: Vec<&TDim> = model.nodes.iter().map(|n| &n.outputs[0].fact.shape[2]).collect();
        for s in 0usize.. {
            let symbols = SymbolValues::default().with(&model.symbol_table.get("S").unwrap(), s as _);
            if dims.iter().all(|d| d.eval(&symbols).to_isize().unwrap() > 0) {
                return s;
            }
        }
        unreachable!();
    }

    pub fn model(ops: &[ConvOp]) -> TypedModel {
        let mut model = InferenceModel::default();
        let s = model.symbol_table.sym("S");
        let mut wire = model.add_source("a", f32::fact(dims!(1, 1, s)).into()).unwrap();
        for (ix, cv) in ops.iter().enumerate() {
            wire = cv.chain(&format!("conv{}", ix), &mut model, wire);
        }
        model.auto_outputs().unwrap();
        model.into_typed().unwrap()
    }

    pub fn run(&self) -> TestCaseResult {
        proptest_regular_against_pulse(
            Self::model(&self.convs),
            self.pulse as _,
            self.input.to_array_view::<f32>().unwrap().to_owned(),
            2,
        )
    }
}

proptest! {
    #[test]
    fn proptest(pb in ConvPlusConvProblem::arbitrary()) { pb.run().unwrap() }
}

fn t(n: usize) -> Tensor {
    tensor1(&(0..n).map(|x| x as f32).collect_vec()).into_shape(&[1, 1, n]).unwrap()
}

#[test]
fn prob_1() {
    let cpc = ConvPlusConvProblem {
        input: t(7),
        pulse: 1,
        convs: vec![
            ConvOp {
                stride: 1,
                dilation: 1,
                ker: tensor3(&[[[1f32]]]),
                padding: PaddingSpec::Valid,
            },
            ConvOp {
                stride: 1,
                dilation: 2,
                ker: tensor3(&[[[1f32, 2.0]]]),
                padding: PaddingSpec::Valid,
            },
        ],
    };
    cpc.run().unwrap();
}

#[test]
fn prob_2() {
    let cpc = ConvPlusConvProblem {
        input: t(10),
        pulse: 2,
        convs: vec![
            ConvOp {
                stride: 2,
                dilation: 1,
                ker: tensor3(&[[[0f32]]]),
                padding: PaddingSpec::SameUpper,
            },
            ConvOp {
                stride: 1,
                dilation: 1,
                ker: tensor3(&[[[1f32]]]),
                padding: PaddingSpec::Valid,
            },
        ],
    };
    cpc.run().unwrap();
}

#[test]
fn prob_3() {
    let cpc = ConvPlusConvProblem {
        input: t(10),
        pulse: 1,
        convs: vec![
            ConvOp {
                stride: 1,
                dilation: 1,
                ker: tensor3(&[[[0f32]]]),
                padding: PaddingSpec::Valid,
            },
            ConvOp {
                stride: 1,
                dilation: 1,
                ker: tensor3(&[[[1f32, 0f32]]]),
                padding: PaddingSpec::SameUpper,
            },
        ],
    };
    cpc.run().unwrap();
}

#[test]
#[ignore]
fn prob_4() {
    let cpc = ConvPlusConvProblem {
        input: t(4),
        pulse: 2,
        convs: vec![
            ConvOp {
                stride: 1,
                dilation: 1,
                ker: tensor3(&[[[0f32]]]),
                padding: PaddingSpec::Valid,
            },
            ConvOp {
                stride: 2,
                dilation: 1,
                ker: tensor3(&[[[0f32, 0f32]]]),
                padding: PaddingSpec::SameUpper,
            },
        ],
    };
    cpc.run().unwrap();
}

#[test]
fn prob_7() {
    let cpc = ConvPlusConvProblem {
        input: t(4),
        pulse: 4,
        convs: vec![
            ConvOp {
                stride: 1,
                dilation: 2,
                ker: tensor3(&[[[0f32, 0.0]]]),
                padding: PaddingSpec::Valid,
            },
            ConvOp {
                stride: 2,
                dilation: 1,
                ker: tensor3(&[[[1f32]]]),
                padding: PaddingSpec::Valid,
            },
        ],
    };
    cpc.run().unwrap();
}

#[test]
fn same_upper() {
    let cpc = ConvPlusConvProblem {
        input: tensor3(&[[[0f32, 0., 0., 1.]]]),
        pulse: 1,
        convs: vec![ConvOp {
            stride: 1,
            dilation: 1,
            ker: tensor3(&[[[1f32, 0.0]]]),
            padding: PaddingSpec::SameUpper,
        }],
    };
    cpc.run().unwrap();
}

#[test]
fn stride() {
    let cpc = ConvPlusConvProblem {
        input: t(4),
        pulse: 2,
        convs: vec![ConvOp {
            stride: 2,
            dilation: 1,
            ker: t(2),
            padding: PaddingSpec::Explicit(tvec!(1), tvec!(0), false),
        }],
    };
    cpc.run().unwrap();
}

#[test]
fn three() {
    let cpc = ConvPlusConvProblem {
        input: t(5),
        pulse: 1,
        convs: vec![
            ConvOp { stride: 1, dilation: 2, ker: t(2), padding: PaddingSpec::Valid },
            ConvOp { stride: 1, dilation: 1, ker: t(3), padding: PaddingSpec::Valid },
            ConvOp {
                stride: 1,
                dilation: 1,
                ker: t(2),
                padding: PaddingSpec::Explicit(tvec!(1), tvec!(0), false),
            },
        ],
    };
    cpc.run().unwrap();
}

#[test]
fn three_stride() {
    let cpc = ConvPlusConvProblem {
        input: t(4),
        pulse: 2,
        convs: vec![ // 0 1 2 3
            ConvOp { stride: 1, dilation: 1, ker: t(2), padding: PaddingSpec::Valid }, // overlap=1, 1 2 3  -> ∂=1
            // pulse: x 1 | 2 3
            ConvOp { stride: 1, dilation: 1, ker: t(1), padding: PaddingSpec::Valid }, // no delay, 0 0 0 -> ∂=1
            // pulse: x 0 | 0 0
            ConvOp { stride: 2, dilation: 2, ker: t(1), padding: PaddingSpec::Valid }, // 0 0
            // pulse 0 | 0
        ],
    };
    cpc.run().unwrap();
}
