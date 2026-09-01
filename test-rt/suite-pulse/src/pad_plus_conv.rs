use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::{Array3, arr3};
use tract_core::ops::array::{Pad, PadMode};
use tract_core::ops::cnn::{Conv, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::runtime::Runtime;

use crate::pulse_and_compare;

#[derive(Debug, Clone)]
pub struct PadPlusConvProblem {
    pub pad_before: usize,
    pub pad_after: usize,
    pub pad_mode: PadMode,
    pub stride: usize,
    pub dilation: usize,
    pub pulse: usize,
    pub ker: Array3<f32>,
    pub input: Array3<f32>,
}

#[derive(Debug, Clone)]
pub struct PadPlusConvProblemParams {
    pub edge: bool,
}

impl Default for PadPlusConvProblemParams {
    fn default() -> Self {
        PadPlusConvProblemParams { edge: true }
    }
}

impl Arbitrary for PadPlusConvProblem {
    type Parameters = PadPlusConvProblemParams;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(params: Self::Parameters) -> BoxedStrategy<PadPlusConvProblem> {
        (
            1usize..3,
            crate::values(1usize..3),
            1usize..3,
            0usize..15,
            0usize..15,
            1usize..3,
            any::<bool>(),
        )
            .prop_flat_map(|(stride, ker, dil, pad_before, pad_after, pulse_factor, edge)| {
                let min_input = (ker.len() * dil).max(pulse_factor * stride);
                (
                    Just(stride),
                    Just(ker),
                    Just(dil),
                    Just(pad_before),
                    Just(pad_after),
                    Just(stride * pulse_factor),
                    crate::values(min_input..3 * min_input),
                    Just(edge),
                )
            })
            .prop_map(move |(stride, ker, dilation, pad_before, pad_after, pulse, input, edge)| {
                let pad_mode = if edge && params.edge && pad_before < pulse {
                    PadMode::Edge
                } else {
                    PadMode::Constant(Tensor::from(9999f32).into())
                };
                let input = Array3::from_shape_vec((1, 1, input.len()), input).unwrap(); // NCHW
                let ker = Array3::from_shape_vec((1, 1, ker.len()), ker).unwrap(); // OIHW
                PadPlusConvProblem {
                    pad_before,
                    pad_after,
                    pad_mode,
                    stride,
                    dilation,
                    pulse,
                    ker,
                    input,
                }
            })
            .boxed()
    }
}

impl Test for PadPlusConvProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let mut wire = model.add_source("a", f32::fact(dims!(1, 1, s)))?;
        if self.pad_before > 0 || self.pad_after > 0 {
            wire = model.wire_node(
                "pad",
                Pad::new(
                    vec![(0, 0), (0, 0), (self.pad_before, self.pad_after)],
                    self.pad_mode.clone(),
                ),
                &[wire],
            )?[0];
        }
        let kernel = model.add_const("kernel", self.ker.clone())?;
        let bias = model.add_const("bias", tensor0(0f32))?;
        let conv = model.wire_node(
            "conv",
            Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: self.ker.shape()[2..].into(),
                    padding: PaddingSpec::Valid,
                    dilations: Some(tvec!(self.dilation)),
                    strides: Some(tvec!(self.stride)),
                    input_channels: 1,
                    output_channels: 1,
                },
                kernel_fmt: tract_core::ops::cnn::KernelFormat::OIHW,
                group: 1,
                q_params: None,
            },
            &[wire, kernel, bias],
        )?;
        model.select_output_outlets(&conv)?;
        pulse_and_compare(runtime, approx, model, self.pulse, self.input.clone().into_dyn(), 2)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<PadPlusConvProblem>("proptest", Default::default());
    suite.add_test(
        "conv_1",
        PadPlusConvProblem {
            pad_before: 0,
            pad_after: 0,
            pad_mode: PadMode::Constant(tensor0(9999f32).into()),
            stride: 1,
            dilation: 1,
            pulse: 1,
            ker: arr3(&[[[0.0f32]]]),
            input: arr3(&[[[0.0f32, 0.0]]]),
        },
    );
    suite.add_test(
        "conv_2",
        PadPlusConvProblem {
            pad_before: 0,
            pad_after: 0,
            pad_mode: PadMode::Constant(tensor0(9999f32).into()),
            stride: 2,
            dilation: 2,
            pulse: 2,
            ker: arr3(&[[[0.0f32]]]),
            input: arr3(&[[[0.0f32, 0.0]]]),
        },
    );
    suite.add_test(
        "conv_3",
        PadPlusConvProblem {
            pad_before: 0,
            pad_after: 0,
            pad_mode: PadMode::Constant(tensor0(9999f32).into()),
            stride: 2,
            dilation: 1,
            pulse: 2,
            ker: arr3(&[[[0.0f32]]]),
            input: arr3(&[[[0.0f32, 0.0, 0.0]]]),
        },
    );
    suite.add_test(
        "conv_4",
        PadPlusConvProblem {
            pad_before: 0,
            pad_after: 0,
            pad_mode: PadMode::Constant(tensor0(9999f32).into()),
            stride: 2,
            dilation: 2,
            pulse: 2,
            ker: arr3(&[[[0.0f32]]]),
            input: arr3(&[[[0.0f32, 0.0, 0.0]]]),
        },
    );
    suite.add_test(
        "conv_5",
        PadPlusConvProblem {
            pad_before: 2,
            pad_after: 0,
            pad_mode: PadMode::Constant(tensor0(9999f32).into()),
            stride: 2,
            dilation: 1,
            pulse: 2,
            ker: arr3(&[[[0.0f32, 1.0]]]),
            input: arr3(&[[[1.0f32, 0.0]]]),
        },
    );
    suite.add_test(
        "conv_6",
        PadPlusConvProblem {
            pad_before: 0,
            pad_after: 0,
            pad_mode: PadMode::Constant(tensor0(9999f32).into()),
            stride: 2,
            dilation: 1,
            pulse: 2,
            ker: arr3(&[[[0.0f32]]]),
            input: arr3(&[[[0.0f32, 0.0, 0.0]]]),
        },
    );
    suite.add_test(
        "conv_7",
        PadPlusConvProblem {
            pad_before: 0,
            pad_after: 1,
            pad_mode: PadMode::Edge,
            stride: 1,
            dilation: 1,
            pulse: 1,
            ker: arr3(&[[[0.0f32]]]),
            input: arr3(&[[[0.0f32]]]),
        },
    );
    suite.add_test(
        "conv_8",
        PadPlusConvProblem {
            pad_before: 1,
            pad_after: 0,
            pad_mode: PadMode::Edge,
            stride: 2,
            dilation: 2,
            pulse: 2,
            ker: arr3(&[[[0.0f32]]]),
            input: arr3(&[[[0.0f32, 0.0f32]]]),
        },
    );
    suite.add_test(
        "conv_kaldi_librispeech",
        PadPlusConvProblem {
            pad_before: 5,
            pad_after: 15,
            pad_mode: PadMode::Edge,
            stride: 3,
            dilation: 1,
            pulse: 9,
            ker: arr3(&[[[1f32, 0f32, 0f32, 0f32, 0f32]]]),
            input: Array3::from_shape_vec((1, 1, 10), (1..=10).map(|i| i as f32).collect())
                .unwrap(),
        },
    );
    suite.add_test(
        "conv_9",
        PadPlusConvProblem {
            pad_before: 13,
            pad_after: 9,
            pad_mode: PadMode::Constant(rctensor0(9999f32)),
            stride: 2,
            dilation: 2,
            pulse: 2,
            ker: arr3(&[[[0.0f32, 0.0]]]),
            input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0]]]),
        },
    );
    Ok(suite)
}
