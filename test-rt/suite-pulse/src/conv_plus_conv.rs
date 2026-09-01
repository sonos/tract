use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::runtime::Runtime;
use tract_core::tract_data::itertools::Itertools;

use crate::pulse_and_compare;

#[derive(Debug, Clone)]
pub struct ConvOp {
    pub stride: usize,
    pub dilation: usize,
    pub ker: Tensor,
    pub padding: PaddingSpec,
}

impl ConvOp {
    fn chain(&self, name: &str, model: &mut TypedModel, after: &[OutletId]) -> TVec<OutletId> {
        let kernel = model.add_const(format!("{name}.k"), self.ker.clone()).unwrap();
        let bias = model.add_const(format!("{name}.b"), tensor0(0f32)).unwrap();
        model
            .wire_node(
                name,
                Conv {
                    pool_spec: PoolSpec {
                        data_format: DataFormat::NCHW,
                        kernel_shape: self.ker.shape()[2..].into(),
                        padding: self.padding.clone(),
                        dilations: Some(tvec!(self.dilation)),
                        strides: Some(tvec!(self.stride)),
                        input_channels: 1,
                        output_channels: 1,
                    },
                    kernel_fmt: KernelFormat::OIHW,
                    group: 1,
                    q_params: None,
                },
                &[after[0], kernel, bias],
            )
            .unwrap()
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
                    PaddingSpec::ExplicitOnnxPool(tvec!(padding - right), tvec!(right), false)
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
pub struct ConvPlusConvProblem {
    pub input: Tensor,
    pub pulse: usize,
    pub convs: Vec<ConvOp>,
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
        let dims: Vec<&TDim> = model
            .nodes
            .iter()
            .filter(|node| !node.outputs[0].fact.shape.is_concrete())
            .map(|n| &n.outputs[0].fact.shape[2])
            .collect();
        for s in 0usize.. {
            let symbols = SymbolValues::default().with(&model.symbols.get("S").unwrap(), s as _);
            if dims.iter().all(|d| d.eval(&symbols).to_isize().unwrap() > 0) {
                return s;
            }
        }
        unreachable!();
    }

    pub fn model(ops: &[ConvOp]) -> TypedModel {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let wire = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
        let mut wire = tvec!(wire);
        for (ix, cv) in ops.iter().enumerate() {
            wire = cv.chain(&format!("conv{ix}"), &mut model, &wire);
        }
        model.select_output_outlets(&wire).unwrap();
        model
    }
}

impl Test for ConvPlusConvProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        pulse_and_compare(
            runtime,
            approx,
            Self::model(&self.convs),
            self.pulse,
            self.input.to_plain_array_view::<f32>()?.to_owned(),
            2,
        )
    }
}

fn t(n: usize) -> Tensor {
    tensor1(&(0..n).map(|x| x as f32).collect_vec()).into_shape(&[1, 1, n]).unwrap()
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<ConvPlusConvProblem>("proptest", ());
    suite.add_test(
        "shrunk_dilation_then_stride",
        ConvPlusConvProblem {
            input: t(3),
            pulse: 2,
            convs: vec![
                ConvOp { stride: 1, dilation: 2, ker: t(2), padding: PaddingSpec::Valid },
                ConvOp {
                    stride: 2,
                    dilation: 2,
                    ker: t(3),
                    padding: PaddingSpec::ExplicitOnnxPool(tvec!(0), tvec!(4), false),
                },
            ],
        },
    );
    suite.add_test(
        "single_conv",
        ConvPlusConvProblem {
            input: tensor3(&[[[1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]]),
            pulse: 4,
            convs: vec![ConvOp {
                stride: 1,
                dilation: 1,
                ker: tensor3(&[[[0.5f32, 1.0, -0.1]]]),
                padding: PaddingSpec::Valid,
            }],
        },
    );
    suite.add_test(
        "prob_1",
        ConvPlusConvProblem {
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
        },
    );
    suite.add_test(
        "prob_2",
        ConvPlusConvProblem {
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
        },
    );
    suite.add_test(
        "prob_3",
        ConvPlusConvProblem {
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
        },
    );
    suite.add_test_with_status(
        "prob_4",
        ConvPlusConvProblem {
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
        },
        infra::TestStatus::Ignored,
    );
    suite.add_test(
        "prob_7",
        ConvPlusConvProblem {
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
        },
    );
    suite.add_test(
        "same_upper",
        ConvPlusConvProblem {
            input: tensor3(&[[[0f32, 0., 0., 1.]]]),
            pulse: 1,
            convs: vec![ConvOp {
                stride: 1,
                dilation: 1,
                ker: tensor3(&[[[1f32, 0.0]]]),
                padding: PaddingSpec::SameUpper,
            }],
        },
    );
    suite.add_test(
        "stride",
        ConvPlusConvProblem {
            input: t(4),
            pulse: 2,
            convs: vec![ConvOp {
                stride: 2,
                dilation: 1,
                ker: t(2),
                padding: PaddingSpec::ExplicitOnnxPool(tvec!(1), tvec!(0), false),
            }],
        },
    );
    suite.add_test(
        "three",
        ConvPlusConvProblem {
            input: t(5),
            pulse: 1,
            convs: vec![
                ConvOp { stride: 1, dilation: 2, ker: t(2), padding: PaddingSpec::Valid },
                ConvOp { stride: 1, dilation: 1, ker: t(3), padding: PaddingSpec::Valid },
                ConvOp {
                    stride: 1,
                    dilation: 1,
                    ker: t(2),
                    padding: PaddingSpec::ExplicitOnnxPool(tvec!(1), tvec!(0), false),
                },
            ],
        },
    );
    suite.add_test(
        "three_stride",
        ConvPlusConvProblem {
            input: t(4),
            pulse: 2,
            convs: vec![
                // 0 1 2 3
                ConvOp { stride: 1, dilation: 1, ker: t(2), padding: PaddingSpec::Valid }, // overlap=1, 1 2 3  -> ∂=1
                // pulse: x 1 | 2 3
                ConvOp { stride: 1, dilation: 1, ker: t(1), padding: PaddingSpec::Valid }, // no delay, 0 0 0 -> ∂=1
                // pulse: x 0 | 0 0
                ConvOp { stride: 2, dilation: 2, ker: t(1), padding: PaddingSpec::Valid }, // 0 0
                                                                                           // pulse 0 | 0
            ],
        },
    );
    Ok(suite)
}
