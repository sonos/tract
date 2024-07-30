use proptest::proptest;
use proptest::test_runner::TestCaseResult;

use super::*;

#[derive(Debug, Clone)]
struct DeconvOp {
    stride: usize,
    dilation: usize,
    adj: usize,
    ker: Array3<f32>,
    padding: PaddingSpec,
}

impl DeconvOp {
    fn chain(&self, name: &str, model: &mut TypedModel, after: OutletId) -> OutletId {
        let deconv = tract_core::ops::cnn::Deconv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(self.ker.shape()[2]),
                padding: self.padding.clone(),
                strides: Some(self.stride).filter(|d| *d > 1).map(|d| tvec!(d)),
                dilations: Some(self.dilation).filter(|d| *d > 1).map(|d| tvec!(d)),
                input_channels: self.ker.shape()[1],
                output_channels: self.ker.shape()[0],
            },
            kernel_format: tract_core::ops::cnn::KernelFormat::OIHW,
            adjustments: tvec!(self.adj),
            group: 1,
        };
        let kernel = model.add_const("kernel", self.ker.clone()).unwrap();
        let bias = model.add_const("bias", rctensor0(0f32)).unwrap();
        model.wire_node(name, deconv, &[after, kernel, bias]).unwrap()[0]
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
                Just(PaddingSpec::Valid),
                Just(PaddingSpec::SameUpper),
                Just(PaddingSpec::SameLower)
            ],
        )
            .prop_filter(
                "Same padding geometry constraint",
                |(stride, dilation, _adj, ker, padding)| {
                    padding == &PaddingSpec::Valid || ((ker.len() - 1) * dilation > stride - 1)
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
        let s = model.symbols.sym("S");
        fact.shape.set(2, s.to_dim());
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
            padding: PaddingSpec::Valid,
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
            padding: PaddingSpec::Valid,
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
            padding: PaddingSpec::Valid,
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
            padding: PaddingSpec::Valid,
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
            padding: PaddingSpec::Valid,
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
            padding: PaddingSpec::SameUpper,
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
            padding: PaddingSpec::Valid,
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
            padding: PaddingSpec::SameUpper,
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
            padding: PaddingSpec::Valid,
        },
    };
    pb.run().unwrap()
}

#[test]
fn deconv2d() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(dims!(1, 2, s, 8))).unwrap();
    let mut kernel = Tensor::zero::<f32>(&[2, 2, 1, 3]).unwrap();
    kernel.as_slice_mut::<f32>().unwrap().iter_mut().enumerate().for_each(|(ix, x)| *x = ix as f32);
    let deconv = tract_core::ops::cnn::Deconv {
        pool_spec: PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec!(1, 3),
            padding: PaddingSpec::Explicit(tvec!(0, 1), tvec!(0, 1)),
            strides: Some(tvec!(1, 2)),
            dilations: Some(tvec![1, 1]),
            input_channels: 2,
            output_channels: 2,
        },
        kernel_format: tract_core::ops::cnn::KernelFormat::OIHW,
        adjustments: tvec!(0, 0),
        group: 1,
    };
    let kernel = model.add_const("kernel", kernel).unwrap();
    let bias = model.add_const("bias", rctensor0(0f32)).unwrap();
    let deconv = model.wire_node("deconv", deconv, &[a, kernel, bias]).unwrap();
    model.set_output_outlets(&deconv).unwrap();
    model.declutter().unwrap();

    let mut input = Tensor::zero::<f32>(&[1, 2, 5, 8]).unwrap();
    input.as_slice_mut::<f32>().unwrap().iter_mut().enumerate().for_each(|(ix, x)| *x = ix as f32);
    proptest_regular_against_pulse(model, 1, input.into_array().unwrap(), 2).unwrap()
}
