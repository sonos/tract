use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use tract_hir::internal::*;
use tract_hir::ops::cnn::{self, PoolSpec};

use super::*;

#[derive(Debug, Clone)]
struct DeconvOp {
    stride: usize,
    dilation: usize,
    adj: usize,
    ker: Array3<f32>,
    padding: cnn::PaddingSpec,
}

impl DeconvOp {
    fn chain(&self, name: &str, model: &mut TypedModel, after: OutletId) -> OutletId {
        let deconv = tract_core::ops::cnn::DeconvUnary {
            pool_spec: PoolSpec {
                data_format: tract_hir::ops::nn::DataFormat::NCHW,
                kernel_shape: tvec!(self.ker.shape()[2]),
                padding: self.padding.clone(),
                strides: Some(self.stride).filter(|d| *d > 1).map(|d| tvec!(d)),
                dilations: Some(self.dilation).filter(|d| *d > 1).map(|d| tvec!(d)),
                output_channel_override: Some(self.ker.shape()[0]),
            },
            kernel_format: tract_core::ops::cnn::KernelFormat::OIHW,
            kernel: self.ker.clone().into_arc_tensor(),
            bias: None,
            adjustments: tvec!(self.adj),
            group: 1,
        };
        model.wire_node(name, deconv, &[after]).unwrap()[0]
    }
}

impl Arbitrary for DeconvOp {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (
            1usize..4,
            1usize..4,
            0usize..4,
            vec(1usize..4),
            prop_oneof![
                Just(cnn::PaddingSpec::Valid),
                Just(cnn::PaddingSpec::SameUpper),
                Just(cnn::PaddingSpec::SameLower)
            ],
        )
            .prop_filter(
                "Same padding geometry constraint",
                |(stride, dilation, _adj, ker, padding)| {
                    padding == &cnn::PaddingSpec::Valid || ((ker.len() - 1) * dilation > stride - 1)
                },
            )
            .prop_map(|(stride, dilation, adj, ker, padding)| DeconvOp {
                stride,
                dilation,
                adj,
                ker: Array3::from_shape_vec((1, 1, ker.len()), ker).unwrap(),
                padding,
            })
            .boxed()
    }
}

#[derive(Debug, Clone)]
struct DeconvProblem {
    input: Array3<f32>,
    pulse: usize,
    deconv: DeconvOp,
}

impl Arbitrary for DeconvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (DeconvOp::arbitrary(), 1usize..3)
            .prop_flat_map(|(deconv, pulse_factor)| {
                let pulse = deconv.stride * pulse_factor;
                let min_input = 4usize;
                (Just(deconv), Just(pulse), vec(min_input..3 * min_input))
            })
            .prop_map(|(deconv, pulse, input)| {
                let input = Array3::from_shape_vec((1, 1, input.len()), input).unwrap(); // NCHW
                DeconvProblem { input, pulse, deconv }
            })
            .boxed()
    }
}

impl DeconvProblem {
    pub fn run(&self) -> TestCaseResult {
        let mut model = TypedModel::default();
        let mut fact = f32::fact(self.input.shape());
        fact.shape.set(2, stream_dim());
        let input = model.add_source("a", fact).unwrap();
        let id = self.deconv.chain("deconv1", &mut model, input);
        model.set_output_outlets(&[id]).unwrap();
        proptest_regular_against_pulse(model, self.pulse as _, self.input.clone().into_dyn(), 2)
    }
}

proptest! {
    #[test]
    fn proptest(pb in DeconvProblem::arbitrary()) { pb.run().unwrap() }
}

#[test]
fn example_0() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 1.0, 0.0]]]),
        pulse: 1,
        deconv: DeconvOp {
            stride: 1,
            dilation: 1,
            adj: 0,
            ker: arr3(&[[[1.0f32]]]),
            padding: cnn::PaddingSpec::Valid,
        },
    };
    pb.run().unwrap()
}

#[test]
fn example_1() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0]]]),
        pulse: 1,
        deconv: DeconvOp {
            stride: 1,
            dilation: 1,
            adj: 0,
            ker: arr3(&[[[0.0f32, 0.0]]]),
            padding: cnn::PaddingSpec::Valid,
        },
    };
    pb.run().unwrap()
}

#[test]
fn example_2() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 0.0, 1.0]]]),
        pulse: 1,
        deconv: DeconvOp {
            stride: 1,
            dilation: 1,
            adj: 0,
            ker: arr3(&[[[0.0f32, 1.0]]]),
            padding: cnn::PaddingSpec::Valid,
        },
    };
    pb.run().unwrap()
}

#[test]
fn example_3() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0, 1.0]]]),
        pulse: 2,
        deconv: DeconvOp {
            stride: 1,
            dilation: 1,
            adj: 0,
            ker: arr3(&[[[0.0f32, 1.0]]]),
            padding: cnn::PaddingSpec::Valid,
        },
    };
    pb.run().unwrap()
}

#[test]
fn dilation_0() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0]]]),
        pulse: 1,
        deconv: DeconvOp {
            stride: 1,
            dilation: 2,
            adj: 0,
            ker: arr3(&[[[0.0f32, 0.0]]]),
            padding: cnn::PaddingSpec::Valid,
        },
    };
    pb.run().unwrap()
}

#[test]
fn dilation_1() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 1.0, 0.0]]]),
        pulse: 1,
        deconv: DeconvOp {
            stride: 1,
            dilation: 2,
            adj: 0,
            ker: arr3(&[[[0.0f32, 1.0]]]),
            padding: cnn::PaddingSpec::SameUpper,
        },
    };
    pb.run().unwrap()
}

#[test]
fn stride_0() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 0.0, 1.0]]]),
        pulse: 2,
        deconv: DeconvOp {
            stride: 2,
            dilation: 1,
            adj: 0,
            ker: arr3(&[[[1.0f32]]]),
            padding: cnn::PaddingSpec::Valid,
        },
    };
    pb.run().unwrap()
}

#[test]
fn same_upper_0() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 0.0, 1.0]]]),
        pulse: 1,
        deconv: DeconvOp {
            stride: 1,
            dilation: 1,
            adj: 0,
            ker: arr3(&[[[0.0f32, 1.0]]]),
            padding: cnn::PaddingSpec::SameUpper,
        },
    };
    pb.run().unwrap()
}

#[test]
fn adj_0() {
    let pb = DeconvProblem {
        input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0]]]),
        pulse: 1,
        deconv: DeconvOp {
            stride: 1,
            dilation: 1,
            adj: 1,
            ker: arr3(&[[[0.0f32]]]),
            padding: cnn::PaddingSpec::Valid,
        },
    };
    pb.run().unwrap()
}
