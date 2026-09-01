use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::{Array3, arr3};
use tract_core::ops::cnn::{Deconv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::runtime::Runtime;

use crate::pulse_and_compare;

#[derive(Debug, Clone)]
pub struct DeconvOp {
    pub stride: usize,
    pub dilation: usize,
    pub adj: usize,
    pub ker: Array3<f32>,
    pub padding: PaddingSpec,
}

impl DeconvOp {
    fn chain(&self, name: &str, model: &mut TypedModel, after: OutletId) -> OutletId {
        let deconv = Deconv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(self.ker.shape()[2]),
                padding: self.padding.clone(),
                strides: Some(self.stride).filter(|d| *d > 1).map(|d| tvec!(d)),
                dilations: Some(self.dilation).filter(|d| *d > 1).map(|d| tvec!(d)),
                input_channels: self.ker.shape()[1],
                output_channels: self.ker.shape()[0],
            },
            kernel_format: KernelFormat::OIHW,
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
            crate::values(1usize..4),
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
pub struct DeconvProblem {
    pub input: Array3<f32>,
    pub pulse: usize,
    pub deconv: DeconvOp,
}

impl Arbitrary for DeconvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (DeconvOp::arbitrary(), 1usize..3)
            .prop_flat_map(|(deconv, pulse_factor)| {
                let pulse = deconv.stride * pulse_factor;
                let min_input = 4usize;
                (Just(deconv), Just(pulse), crate::values(min_input..3 * min_input))
            })
            .prop_map(|(deconv, pulse, input)| {
                let input = Array3::from_shape_vec((1, 1, input.len()), input).unwrap(); // NCHW
                DeconvProblem { input, pulse, deconv }
            })
            .boxed()
    }
}

impl Test for DeconvProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let mut fact = f32::fact(self.input.shape());
        let s = model.symbols.sym("S");
        fact.shape.set(2, s.to_dim());
        let input = model.add_source("a", fact).unwrap();
        let id = self.deconv.chain("deconv1", &mut model, input);
        model.select_output_outlets(&[id]).unwrap();
        pulse_and_compare(runtime, approx, model, self.pulse, self.input.clone().into_dyn(), 2)
    }
}

#[derive(Clone, Debug)]
pub struct Deconv2d;

impl Test for Deconv2d {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, 2, s, 8))).unwrap();
        let mut kernel = Tensor::zero::<f32>(&[2, 2, 1, 3]).unwrap();
        kernel
            .try_as_plain_mut()
            .unwrap()
            .as_slice_mut::<f32>()
            .unwrap()
            .iter_mut()
            .enumerate()
            .for_each(|(ix, x)| *x = ix as f32);
        let deconv = Deconv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(1, 3),
                padding: PaddingSpec::Explicit(tvec!(0, 1), tvec!(0, 1)),
                strides: Some(tvec!(1, 2)),
                dilations: Some(tvec![1, 1]),
                input_channels: 2,
                output_channels: 2,
            },
            kernel_format: KernelFormat::OIHW,
            adjustments: tvec!(0, 0),
            group: 1,
        };
        let kernel = model.add_const("kernel", kernel).unwrap();
        let bias = model.add_const("bias", rctensor0(0f32)).unwrap();
        let deconv = model.wire_node("deconv", deconv, &[a, kernel, bias]).unwrap();
        model.select_output_outlets(&deconv).unwrap();
        model.declutter().unwrap();

        let mut input = Tensor::zero::<f32>(&[1, 2, 5, 8]).unwrap();
        input
            .try_as_plain_mut()
            .unwrap()
            .as_slice_mut::<f32>()
            .unwrap()
            .iter_mut()
            .enumerate()
            .for_each(|(ix, x)| *x = ix as f32);
        pulse_and_compare(runtime, approx, model, 1, input.into_plain_array()?, 2)
    }
}

// Issue #2203: pulse-mode Deconv with non-zero bias and kernel > stride
// double-counts the bias in the overlap region. Bulk adds bias once per
// output slot; the per-pulse Deconv also adds bias to its full
// ``P*S + (K-1)`` output, and the DeconvDelay overlap-add then sums
// pulse N's bias-included tail into pulse N+1's bias-included head.
// Surfaced by Pocket-TTS / Mimi (depthwise ConvTranspose1d, K=32, S=16).
// Existing ``proptest`` and ``example_*`` cases all use ``bias=0`` (see
// ``DeconvOp::chain``), which masks the bug.

#[derive(Clone, Debug)]
pub struct Issue2203 {
    pub group: usize,
    pub output_channels: usize,
    pub bias: tract_core::ndarray::Array1<f32>,
}

impl Test for Issue2203 {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let (group, output_channels, bias) = (self.group, self.output_channels, self.bias.clone());
        let ker_len = 32usize;
        let stride = 16usize;
        let pulse = 2usize;
        let input_len = 8usize;
        let in_channels_per_group = output_channels / group;

        let mut model = TypedModel::default();
        let mut fact = f32::fact([1, output_channels, input_len]);
        let s = model.symbols.sym("S");
        fact.shape.set(2, s.to_dim());
        let input = model.add_source("a", fact).unwrap();
        let kernel = tract_ndarray::Array3::from_shape_vec(
            (output_channels, in_channels_per_group, ker_len),
            (0..output_channels * in_channels_per_group * ker_len)
                .map(|i| 0.001_f32 * (i as f32 + 1.0))
                .collect(),
        )
        .unwrap();
        let deconv = Deconv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(ker_len),
                padding: PaddingSpec::Explicit(tvec!(0), tvec!(0)),
                strides: Some(tvec!(stride)),
                dilations: None,
                input_channels: output_channels,
                output_channels,
            },
            kernel_format: KernelFormat::OIHW,
            adjustments: tvec!(0),
            group,
        };
        let kernel_node = model.add_const("kernel", kernel).unwrap();
        let bias_node = model.add_const("bias", bias).unwrap();
        let id = model.wire_node("deconv1", deconv, &[input, kernel_node, bias_node]).unwrap()[0];
        model.select_output_outlets(&[id]).unwrap();

        let input = tract_ndarray::Array3::from_shape_vec(
            (1, output_channels, input_len),
            (0..output_channels * input_len).map(|i| 0.1_f32 * (i as f32 + 1.0)).collect(),
        )
        .unwrap();
        pulse_and_compare(runtime, approx, model, pulse, input.into_dyn(), 2)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<DeconvProblem>("proptest", ());
    suite.add_test("deconv2d", Deconv2d);
    suite.add_test(
        "example_0",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 1.0, 0.0]]]),
            pulse: 1,
            deconv: DeconvOp {
                stride: 1,
                dilation: 1,
                adj: 0,
                ker: arr3(&[[[1.0f32]]]),
                padding: PaddingSpec::Valid,
            },
        },
    );
    suite.add_test(
        "example_1",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0]]]),
            pulse: 1,
            deconv: DeconvOp {
                stride: 1,
                dilation: 1,
                adj: 0,
                ker: arr3(&[[[0.0f32, 0.0]]]),
                padding: PaddingSpec::Valid,
            },
        },
    );
    suite.add_test(
        "example_2",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 0.0, 1.0]]]),
            pulse: 1,
            deconv: DeconvOp {
                stride: 1,
                dilation: 1,
                adj: 0,
                ker: arr3(&[[[0.0f32, 1.0]]]),
                padding: PaddingSpec::Valid,
            },
        },
    );
    suite.add_test(
        "example_3",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0, 1.0]]]),
            pulse: 2,
            deconv: DeconvOp {
                stride: 1,
                dilation: 1,
                adj: 0,
                ker: arr3(&[[[0.0f32, 1.0]]]),
                padding: PaddingSpec::Valid,
            },
        },
    );
    suite.add_test(
        "dilation_0",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0]]]),
            pulse: 1,
            deconv: DeconvOp {
                stride: 1,
                dilation: 2,
                adj: 0,
                ker: arr3(&[[[0.0f32, 0.0]]]),
                padding: PaddingSpec::Valid,
            },
        },
    );
    suite.add_test(
        "dilation_1",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 1.0, 0.0]]]),
            pulse: 1,
            deconv: DeconvOp {
                stride: 1,
                dilation: 2,
                adj: 0,
                ker: arr3(&[[[0.0f32, 1.0]]]),
                padding: PaddingSpec::SameUpper,
            },
        },
    );
    suite.add_test(
        "stride_0",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 0.0, 1.0]]]),
            pulse: 2,
            deconv: DeconvOp {
                stride: 2,
                dilation: 1,
                adj: 0,
                ker: arr3(&[[[1.0f32]]]),
                padding: PaddingSpec::Valid,
            },
        },
    );
    suite.add_test(
        "same_upper_0",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 0.0, 1.0]]]),
            pulse: 1,
            deconv: DeconvOp {
                stride: 1,
                dilation: 1,
                adj: 0,
                ker: arr3(&[[[0.0f32, 1.0]]]),
                padding: PaddingSpec::SameUpper,
            },
        },
    );
    suite.add_test(
        "adj_0",
        DeconvProblem {
            input: arr3(&[[[0.0f32, 0.0, 0.0, 0.0]]]),
            pulse: 1,
            deconv: DeconvOp {
                stride: 1,
                dilation: 1,
                adj: 1,
                ker: arr3(&[[[0.0f32]]]),
                padding: PaddingSpec::Valid,
            },
        },
    );
    suite.add_test(
        "issue_2203_dense_with_bias_kernel_32_stride_16",
        Issue2203 {
            group: 1,
            output_channels: 8,
            bias: tract_core::ndarray::Array1::<f32>::from_elem((8,), 0.5_f32),
        },
    );
    // Mimi's actual configuration: depthwise (groups = output_channels),
    // kernel (G, 1, K) in OIHW.
    suite.add_test(
        "issue_2203_depthwise_with_bias_kernel_32_stride_16",
        Issue2203 {
            group: 8,
            output_channels: 8,
            bias: tract_core::ndarray::Array1::<f32>::from_shape_vec(
                (8,),
                (0..8).map(|i| 0.001_f32 * (i as f32 + 1.0)).collect(),
            )?,
        },
    );
    Ok(suite)
}
