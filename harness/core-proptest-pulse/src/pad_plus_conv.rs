use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use proptest::*;

use super::*;

#[derive(Debug, Clone)]
struct PadPlusConvProblem {
    pad_before: usize,
    pad_after: usize,
    pad_mode: PadMode,
    stride: usize,
    dilation: usize,
    pulse: usize,
    ker: Array3<f32>,
    input: Array3<f32>,
}

impl Arbitrary for PadPlusConvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<PadPlusConvProblem> {
        (1usize..3, vec(1usize..3), 1usize..3, 0usize..15, 0usize..15, 1usize..3, bool::ANY)
            .prop_flat_map(|(stride, ker, dil, pad_before, pad_after, pulse_factor, edge)| {
                let min_input = (ker.len() * dil).max(pulse_factor * stride);
                (
                    Just(stride),
                    Just(ker),
                    Just(dil),
                    Just(pad_before),
                    Just(pad_after),
                    Just(stride * pulse_factor),
                    vec(min_input..3 * min_input),
                    Just(edge),
                )
            })
            .prop_map(|(stride, ker, dilation, pad_before, pad_after, pulse, input, edge)| {
                let pad_mode = if edge && pad_before < pulse {
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

impl PadPlusConvProblem {
    pub fn run(&self) -> TestCaseResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let mut wire = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
        if self.pad_before > 0 || self.pad_after > 0 {
            wire = model
                .wire_node(
                    "pad",
                    Pad::new(
                        vec![(0, 0), (0, 0), (self.pad_before, self.pad_after)],
                        self.pad_mode.clone(),
                    ),
                    &[wire],
                )
                .unwrap()[0];
        }
        let kernel = model.add_const("kernel", self.ker.clone()).unwrap();
        let bias = model.add_const("bias", tensor0(0f32)).unwrap();
        let conv = model
            .wire_node(
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
            )
            .unwrap();
        model.set_output_outlets(&conv).unwrap();
        proptest_regular_against_pulse(model, self.pulse as _, self.input.clone().into_dyn(), 2)
    }
}

proptest! {
    #[test]
    fn proptest_conv(pb in PadPlusConvProblem::arbitrary()) { pb.run().unwrap() }
}

#[test]
fn conv_1() {
    PadPlusConvProblem {
        pad_before: 0,
        pad_after: 0,
        pad_mode: PadMode::Constant(tensor0(9999f32).into()),
        stride: 1,
        dilation: 1,
        pulse: 1,
        ker: arr3(&[[[0.0f32]]]),
        input: arr3(&[[[0.0f32, 0.0]]]),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_2() {
    PadPlusConvProblem {
        pad_before: 0,
        pad_after: 0,
        pad_mode: PadMode::Constant(tensor0(9999f32).into()),
        stride: 2,
        dilation: 2,
        pulse: 2,
        ker: arr3(&[[[0.0f32]]]),
        input: arr3(&[[[0.0f32, 0.0]]]),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_3() {
    PadPlusConvProblem {
        pad_before: 0,
        pad_after: 0,
        pad_mode: PadMode::Constant(tensor0(9999f32).into()),
        stride: 2,
        dilation: 1,
        pulse: 2,
        ker: arr3(&[[[0.0f32]]]),
        input: arr3(&[[[0.0f32, 0.0, 0.0]]]),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_4() {
    PadPlusConvProblem {
        pad_before: 0,
        pad_after: 0,
        pad_mode: PadMode::Constant(tensor0(9999f32).into()),
        stride: 2,
        dilation: 2,
        pulse: 2,
        ker: arr3(&[[[0.0f32]]]),
        input: arr3(&[[[0.0f32, 0.0, 0.0]]]),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_5() {
    PadPlusConvProblem {
        pad_before: 2,
        pad_after: 0,
        pad_mode: PadMode::Constant(tensor0(9999f32).into()),
        stride: 2,
        dilation: 1,
        pulse: 2,
        ker: arr3(&[[[0.0f32, 1.0]]]),
        input: arr3(&[[[1.0f32, 0.0]]]),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_6() {
    PadPlusConvProblem {
        pad_before: 0,
        pad_after: 0,
        pad_mode: PadMode::Constant(tensor0(9999f32).into()),
        stride: 2,
        dilation: 1,
        pulse: 2,
        ker: arr3(&[[[0.0f32]]]),
        input: arr3(&[[[0.0f32, 0.0, 0.0]]]),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_7() {
    PadPlusConvProblem {
        pad_before: 0,
        pad_after: 1,
        pad_mode: PadMode::Edge,
        stride: 1,
        dilation: 1,
        pulse: 1,
        ker: arr3(&[[[0.0f32]]]),
        input: arr3(&[[[0.0f32]]]),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_8() {
    PadPlusConvProblem {
        pad_before: 1,
        pad_after: 0,
        pad_mode: PadMode::Edge,
        stride: 2,
        dilation: 2,
        pulse: 2,
        ker: arr3(&[[[0.0f32]]]),
        input: arr3(&[[[0.0f32, 0.0f32]]]),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_kaldi_librispeech() {
    PadPlusConvProblem {
        pad_before: 5,
        pad_after: 15,
        pad_mode: PadMode::Edge,
        stride: 3,
        dilation: 1,
        pulse: 9,
        ker: arr3(&[[[1f32, 0f32, 0f32, 0f32, 0f32]]]),
        input: Array3::from_shape_vec((1, 1, 10), (1..=10).map(|i| i as f32).collect()).unwrap(),
    }
    .run()
    .unwrap()
}

#[test]
fn conv_9() {
    PadPlusConvProblem {
        pad_before: 13,
        pad_after: 9,
        pad_mode: PadMode::Constant(rctensor0(9999f32)),
        stride: 2,
        dilation: 2,
        pulse: 2,
        ker: arr3(&[[[0.0f32, 0.0]]]),
        input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0]]]),
    }
    .run()
    .unwrap()
}
