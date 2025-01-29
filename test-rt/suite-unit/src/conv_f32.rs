use std::ops::Range;

use super::*;
use infra::*;
use proptest::collection::vec;
use tract_itertools::izip;

#[derive(Debug, Clone, Default)]
pub struct ConvProblemParams {
    pub no_group: bool,
    pub no_arbitrary_grouping: bool,
    pub geo_rank: Option<Range<usize>>,
    pub no_batch: bool,
}

#[derive(Debug, Clone)]
pub struct ConvProblem {
    pub shape_in: DataShape,
    pub kernel_format: KernelFormat,
    pub group: usize,
    pub data: ArrayD<f32>,
    pub kernel: ArrayD<f32>,
    pub bias: Option<ArrayD<f32>>,
    pub pad: PaddingSpec,
    pub strides: TVec<usize>,
}

impl ConvProblem {
    fn geo_ker(&self) -> &[usize] {
        &self.kernel.shape()[self.kernel_format.h_axis()..][..self.shape_in.hw_rank()]
    }

    pub fn reference(&self) -> ArrayD<f32> {
        assert_eq!(self.data.shape(), &*self.shape_in.shape, "inconsistent shapes in test");
        let n = *self.shape_in.n().unwrap_or(&1);
        let ci_per_g = self.shape_in.c() / self.group;
        let co_per_g = match self.kernel_format {
            KernelFormat::OIHW => self.kernel.shape()[0] / self.group,
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1],
            KernelFormat::OHWI => self.kernel.shape()[0],
        };
        assert_eq!(self.strides.len(), self.geo_ker().len());
        let (shape_out, left_pads): (TVec<_>, TVec<_>) = match &self.pad {
            PaddingSpec::Valid => izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides)
                .map(|(i, k, s)| {
                    let out = (*i + 1).saturating_sub(*k).divceil(*s);
                    (out, 0)
                })
                .unzip(),
            PaddingSpec::SameUpper => izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides)
                .map(|(input, k, stride)| {
                    let out = input.divceil(*stride);
                    let pad = ((out - 1) * stride + k).saturating_sub(*input);
                    (out, pad / 2)
                })
                .unzip(),
            PaddingSpec::SameLower => izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides)
                .map(|(input, k, stride)| {
                    let out = input.divceil(*stride);
                    let pad = ((out - 1) * stride + k).saturating_sub(*input);
                    (out, pad.divceil(2))
                })
                .unzip(),
            PaddingSpec::Explicit(l, r) => {
                izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides, l, r)
                    .map(|(input, k, stride, l, r)| {
                        let dil = 1;
                        let kf = (k - 1) * dil + 1;
                        let out = (input + l + r).saturating_sub(kf) / *stride + 1;
                        (out, *l)
                    })
                    .unzip()
            }
            PaddingSpec::ExplicitOnnxPool(l, r, ceil) => {
                izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides, l, r)
                    .map(|(input, k, stride, l, r)| {
                        let dil = 1;
                        let kf = (k - 1) * dil + 1;
                        let mut out = if *ceil {
                            (input + l + r).saturating_sub(kf).divceil(*stride) + 1
                        } else {
                            (input + l + r).saturating_sub(kf) / *stride + 1
                        };

                        // pytorch semantics diverge from onnx (and onnx are super weird)
                        // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Pool.h#L48C2-L54C6
                        if *ceil && (out - 1) * stride >= input + l {
                            out -= 1;
                        }
                        (out, *l)
                    })
                    .unzip()
            }
        };
        let shape_out = self
            .shape_in
            .fmt
            .from_n_c_hw(self.shape_in.n().cloned().unwrap_or(1), co_per_g * self.group, shape_out)
            .unwrap();
        let mut out = ArrayD::zeros(&*shape_out.shape);
        for n in 0..n {
            for g in 0..self.group {
                for geo_out in indices(shape_out.hw_dims()) {
                    let mut output_coords: TVec<usize> = geo_out.slice().into();
                    if self.shape_in.fmt.has_n() {
                        output_coords.insert(0, n);
                    }
                    output_coords.insert(shape_out.c_axis(), 0);
                    for geo_ker in indices(self.geo_ker()) {
                        let input_coords: TVec<isize> =
                            izip!(geo_out.slice(), geo_ker.slice(), &left_pads, &self.strides)
                                .map(|(out, ker, pad, stride)| {
                                    *out as isize * *stride as isize + *ker as isize - *pad as isize
                                })
                                .collect();
                        if izip!(&input_coords, self.shape_in.hw_dims())
                            .any(|(c, i)| *c < 0 || *c >= *i as isize)
                        {
                            continue;
                        }
                        let mut input_coords: TVec<usize> =
                            input_coords.into_iter().map(|d| d as usize).collect();
                        if self.shape_in.fmt.has_n() {
                            input_coords.insert(0, n);
                        }
                        input_coords.insert(self.shape_in.c_axis(), 0);
                        for ci in 0..ci_per_g {
                            input_coords[self.shape_in.c_axis()] = ci + g * ci_per_g;
                            let i = self.data[&*input_coords];
                            for co in 0..co_per_g {
                                output_coords[shape_out.c_axis()] = co + g * co_per_g;
                                let mut kernel_coords: TVec<usize> = geo_ker.slice().into();
                                match self.kernel_format {
                                    KernelFormat::OIHW => {
                                        kernel_coords.insert(0, ci);
                                        kernel_coords.insert(0, co + g * co_per_g);
                                    }
                                    KernelFormat::HWIO => {
                                        kernel_coords.push(ci + g * ci_per_g);
                                        kernel_coords.push(co);
                                    }
                                    KernelFormat::OHWI => {
                                        kernel_coords.insert(0, co);
                                        kernel_coords.push(ci + g * ci_per_g);
                                    }
                                }
                                let k = self.kernel[&*kernel_coords];
                                out[&*output_coords] += k * i;
                            }
                        }
                    }
                }
            }
        }
        if let Some(bias) = &self.bias {
            let mut shape = vec![1; out.ndim()];
            shape[shape_out.c_axis()] = bias.len();
            out += &bias.clone().into_shape_with_order(shape).unwrap();
        }
        out
    }

    pub fn tract(&self) -> TractResult<TypedModel> {
        assert_eq!(self.data.shape(), &*self.shape_in.shape, "inconsistent shapes in test");
        let mut model = TypedModel::default();
        let wire = model.add_source("input", f32::fact(&self.shape_in.shape))?;
        let ci = *self.shape_in.c();
        let co = match self.kernel_format {
            KernelFormat::OIHW => self.kernel.shape()[0],
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1] * self.group,
            KernelFormat::OHWI => self.kernel.shape()[0] * self.group,
        };
        let kernel = model.add_const("kernel", self.kernel.clone().into_arc_tensor())?;
        let bias = if let Some(bias) = &self.bias {
            bias.clone().into_arc_tensor()
        } else {
            rctensor0(0f32)
        };
        let bias = model.add_const("bias", bias)?;
        let op = Conv::new(
            PoolSpec::new(
                self.shape_in.fmt,
                self.geo_ker().into(),
                self.pad.clone(),
                None,
                Some(self.strides.clone()),
                ci,
                co,
            ),
            self.kernel_format,
            self.group,
            None,
        );
        let wire = model.wire_node("conv", op, &[wire, kernel, bias])?[0];
        model.set_output_outlets(&[wire])?;
        Ok(model)
    }
}

impl Arbitrary for ConvProblem {
    type Parameters = ConvProblemParams;
    type Strategy = BoxedStrategy<ConvProblem>;
    fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
        let batch_range = if params.no_batch { 1usize..=1 } else { 1usize..=3 };
        let geo_rank = params.geo_rank.unwrap_or(1..4);
        (
            data_format(),
            kernel_format(),
            prop_oneof![Just(PaddingSpec::Valid), Just(PaddingSpec::SameUpper)],
            batch_range,
            1usize..=4,
            1usize..=4,
            1usize..=(if params.no_group { 1 } else { 3 }),
            geo_rank.prop_flat_map(shapes),
        )
            .prop_flat_map(
                move |(
                    df,
                    kf,
                    pad,
                    batch,
                    mut ci0,
                    mut co0,
                    group,
                    (mut ker_shape, data_shape),
                )| {
                    // FIXME in HWIO order, only regular and depthwise are supported
                    if params.no_arbitrary_grouping && group > 1 {
                        ci0 = 1;
                        co0 = 1;
                    }
                    if kf == KernelFormat::HWIO && group > 1 {
                        ci0 = 1;
                    }
                    let shape_in = df.from_n_c_hw(batch, ci0 * group, data_shape).unwrap();
                    let data_in = tensor(&*shape_in.shape);
                    match kf {
                        KernelFormat::HWIO => {
                            ker_shape.push(ci0 * group);
                            ker_shape.push(co0)
                        }
                        KernelFormat::OIHW => {
                            ker_shape.insert(0, ci0);
                            ker_shape.insert(0, co0 * group)
                        }
                        KernelFormat::OHWI => {
                            ker_shape.insert(0, co0);
                            ker_shape.push(ci0 * group);
                        }
                    };
                    let strides = vec(1usize..=3, shape_in.hw_rank()..=shape_in.hw_rank());
                    let kernel = tensor(&ker_shape);
                    let bias = proptest::option::of(tensor(&[co0 * group]));
                    (Just((kf, pad, shape_in, group)), data_in, kernel, bias, strides)
                },
            )
            .prop_map(|((kernel_format, pad, shape_in, group), data, kernel, bias, strides)| {
                ConvProblem {
                    shape_in,
                    kernel_format,
                    group,
                    data,
                    kernel,
                    bias,
                    pad,
                    strides: strides.into(),
                }
            })
            .boxed()
    }
}

impl Test for ConvProblem {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        // dbg!(&reference);
        let mut model = self.tract()?;
        //       dbg!(&model);
        model.declutter()?;
        //       dbg!(&model);
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut output = runtime.prepare(model)?.run(tvec![self.data.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<ConvProblem>("proptest", ConvProblemParams::default());

    suite.add(
        "trivial_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0f32]]]).into_dyn(),
            kernel: arr4(&[[[[0.0f32]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1),
        },
    );

    suite.add(
        "trivial_1",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(1, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[1.0f32]]]).into_dyn(),
            kernel: arr3(&[[[1.0f32]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );
    suite.add(
        "trivial_2",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[1.0f32], [0.0]]]).into_dyn(),
            kernel: arr3(&[[[1.0f32]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "trivial_3",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
            kernel: arr3(&[[[0.0f32], [1.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "nchw_0",
        ConvProblem {
            shape_in: DataFormat::NCHW.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0f32, 1.0]]]).into_dyn(),
            kernel: arr3(&[[[1f32]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_0",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr2(&[[0.0f32], [0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0f32]], [[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr2(&[[0.0f32, 1.0]]).into_dyn(),
            kernel: arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_3",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr2(&[[0.0f32, 1.0]]).into_dyn(),
            kernel: arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );
    suite.add(
        "group_4",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr2(&[[0.0f32, 1.0]]).into_dyn(),
            kernel: arr3(&[[[0.0f32]], [[0.0]], [[0.0]], [[1.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );
    suite.add(
        "group_5",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1, 1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
            kernel: tensor4(&[[[[0.0f32]]], [[[0.0]]], [[[0.0]]], [[[0.0]]]])
                .into_array::<f32>()
                .unwrap()
                .into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1),
        },
    );
    suite.add(
        "group_6",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
            kernel: tensor3(&[[[0.0f32]], [[0.0]], [[0.0]], [[0.0]]])
                .into_array::<f32>()
                .unwrap()
                .into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_7",
        ConvProblem {
            shape_in: DataFormat::NCHW.from_n_c_hw(1, 2, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr3(&[[[0.0f32, 0.0], [0.0, 1.0]]]).into_dyn(),
            kernel: tensor3(&[[[0.0f32, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 1.0]]])
                .into_array::<f32>()
                .unwrap()
                .into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );
    suite.add(
        "group_8",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 4, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr2(&[[0.0f32, 0.0, 0.0, 1.0]]).into_dyn(),
            kernel: tensor3(&[[[0.0f32], [0.0]], [[0.0], [0.0]]])
                .into_array::<f32>()
                .unwrap()
                .into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_9",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr2(&[[0.0f32, 0.0], [0.0, 1.0]]).into_dyn(),
            kernel: tensor3(&[[[0.0f32]], [[0.0]], [[0.0]], [[1.0]]])
                .into_array::<f32>()
                .unwrap()
                .into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );
    suite.add(
        "group_10",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [2, 1, 4])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: ArrayD::<f32>::zeros(vec![2, 2, 1, 4]),
            kernel: ArrayD::from_elem(vec![4, 1, 1, 1, 2], 1.0f32),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1, 1),
        },
    );
    suite.add(
        "group_11",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr2(&[[0.0, 1.0]]).into_dyn(),
            kernel: arr3(&[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.0]]])
                .into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_12",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::HWIO,
            group: 2,
            data: arr2(&[[0.0, 0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0], [0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_13",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::HWIO,
            group: 2,
            data: arr2(&[[0.0, 1.0]]).into_dyn(),
            kernel: arr3(&[[[0.0, 0.0], [1.0, 0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_bias_0",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: ArrayD::<f32>::zeros(vec![1, 1, 2]),
            kernel: ArrayD::<f32>::zeros(vec![4, 1, 1]),
            bias: Some(ArrayD::<f32>::zeros(vec![4])),
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "group_bias_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 2,
            data: arr2(&[[0.0, 0.0]]).into_dyn(),
            kernel: ArrayD::<f32>::zeros(vec![4, 1, 1]),
            bias: Some(arr1(&[0.0, 0.0, 0.0, 1.0]).into_dyn()),
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "bias_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![2, 1]),
            kernel: ArrayD::<f32>::zeros(vec![1, 1, 2]),
            bias: Some(ArrayD::<f32>::zeros(vec![1])),
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "bias_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            kernel: ArrayD::<f32>::zeros(vec![2, 1, 2]),
            data: ArrayD::<f32>::zeros(vec![2, 1]),
            bias: Some(arr1(&[0.0f32, 1.0]).into_dyn()),
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "bias_chw_0",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr2(&[[0f32, 0., 0.]]).into_dyn(),
            kernel: arr3(&[[[0f32]], [[0.]], [[0.]]]).into_dyn(),
            bias: Some(arr1(&[0f32, 0., 1.]).into_dyn()),
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "batch_0",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(2, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![2, 2, 1]),
            kernel: ArrayD::<f32>::zeros(vec![1, 1, 2]),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "batch_1",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(2, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
            kernel: arr3(&[[[1.0f32]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "batch_2",
        ConvProblem {
            shape_in: DataFormat::NCHW.from_n_c_hw(2, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0f32, 0.0]], [[0.0, 0.0]]]).into_dyn(),
            kernel: arr3(&[[[0.0f32]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "bias_3d_1",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [1, 1, 2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![1, 1, 1, 2]),
            kernel: ArrayD::<f32>::zeros(vec![1, 1, 1, 1, 1]),
            bias: Some(ArrayD::<f32>::ones(vec![1])),
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1, 1),
        },
    );

    suite.add(
        "batch_3d",
        ConvProblem {
            shape_in: DataFormat::NCHW.from_n_c_hw(1, 1, [2, 2, 1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![1, 1, 2, 2, 1]),
            kernel: ArrayD::<f32>::zeros(vec![1, 1, 1, 1, 1]),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1, 1),
        },
    );

    suite.add(
        "same_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![1, 1]),
            kernel: ArrayD::<f32>::zeros(vec![1, 1, 1]),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1),
        },
    );

    suite.add(
        "same_1d_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![1, 1]),
            kernel: ArrayD::<f32>::zeros(vec![1, 1, 1]),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1),
        },
    );

    suite.add(
        "same_1d_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr2(&[[0.0], [1.0]]).into_dyn(),
            kernel: arr3(&[[[0.0, 2.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1),
        },
    );

    suite.add(
        "same_2",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0], [0.0]], [[0.0], [1.0]]]).into_dyn(),
            kernel: arr4(&[[[[0.0], [0.0]]], [[[0.0], [1.0]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 1),
        },
    );

    suite.add(
        "same_2d_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0], [0.0], [1.0]]]).into_dyn(),
            kernel: arr4(&[[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1.0]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 1),
        },
    );

    suite.add(
        "same_2d_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0], [0.0]], [[1.0], [0.0]]]).into_dyn(),
            kernel: arr4(&[[[[0.0, 1.0]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 1),
        },
    );

    suite.add(
        "same_2d_3",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [2, 3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]).into_dyn(),
            kernel: arr4(&[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 1),
        },
    );

    suite.add(
        "strides_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr2(&[[0.0], [0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(2),
        },
    );

    suite.add(
        "strides_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr2(&[[0.0], [0.0], [1.0]]).into_dyn(),
            kernel: arr3(&[[[1.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(2),
        },
    );

    suite.add(
        "strides_2_dnn_padding_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [6])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::arr2(&[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]).into_dyn(),
            kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(1), tvec!(1)),
            strides: tvec!(2),
        },
    );

    suite.add(
        "strides_2_dnn_padding_2",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::arr2(&[[0.0]]).into_dyn(),
            kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(1), tvec!(1)),
            strides: tvec!(2),
        },
    );

    suite.add(
        "strides_2_dnn_padding_ceil_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::arr2(&[[0.0]]).into_dyn(),
            kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(0), tvec!(1)),
            strides: tvec!(2),
        },
    );

    suite.add(
        "strides_2_dnn_padding_ceil_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::arr2(&[[0.0], [0.0]]).into_dyn(),
            kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(0), tvec!(0)),
            strides: tvec!(2),
        },
    );

    suite.add(
        "strides_2",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr2(&[[0.0], [0.0], [1.0]]).into_dyn(),
            kernel: arr3(&[[[0.0, 0.0, 1.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(3),
        },
    );

    suite.add(
        "strides_2d_same_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0], [0.0], [1.0]]]).into_dyn(),
            kernel: arr4(&[[[[1.0, 0.0]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 2),
        },
    );

    suite.add(
        "strides_2d_same_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![2, 3, 1]),
            kernel: ArrayD::<f32>::zeros(vec![1, 1, 1, 3]),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 2),
        },
    );

    suite.add(
        "strides_2d_same_2",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0], [0.0], [1.0]], [[0.0], [0.0], [0.0]]]).into_dyn(),
            kernel: arr4(&[[[[1.0, 0.0, 0.0]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 2),
        },
    );

    suite.add(
        "strides_two_axes",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0.0]]]).into_dyn(),
            kernel: arr4(&[[[[0.0]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(2, 2),
        },
    );

    suite.add(
        "strides_one_axis_explicit",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::ArrayD::<f32>::zeros(vec![3, 1]),
            kernel: tract_ndarray::ArrayD::<f32>::zeros(vec![1, 1, 3]),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(1), tvec!(0)),
            strides: tvec!(2),
        },
    );

    suite.add(
        "strides_one_axis_explicit_prime",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::ArrayD::<f32>::zeros(vec![2, 1]),
            kernel: tract_ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2]),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(0), tvec!(0)),
            strides: tvec!(1),
        },
    );

    suite.add(
        "strides_two_axes_explicit",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 3])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::ArrayD::<f32>::zeros(vec![2, 3, 1]),
            kernel: tract_ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2, 3]),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(0, 1), tvec!(0, 0)),
            strides: tvec!(1, 2),
        },
    );

    suite.add(
        "lazy_im2col_0",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [2])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr2(&[[0.0, 0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0, 0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    let mut kernel = ArrayD::<f32>::zeros(vec![1, 4, 1, 3, 2]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    suite.add(
        "lazy_im2col_big",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 4, [2, 5, 4])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![4, 2, 5, 4]),
            kernel,
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(2, 2, 2),
        },
    );

    let mut kernel = ArrayD::<f32>::zeros(vec![1, 4, 1, 3, 2]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    suite.add(
        "lazy_im2col_big_2",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(1, 4, [2, 5, 4])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::<f32>::zeros(vec![1, 2, 5, 4, 4]),
            kernel,
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(2, 3, 2),
        },
    );

    let mut kernel = ArrayD::<f32>::zeros(vec![2, 2, 2, 1]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    let mut data = ArrayD::<f32>::zeros(vec![2, 2, 2]);
    *data.as_slice_mut().unwrap().last_mut().unwrap() = 1.0;
    suite.add(
        "depthwise_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [2, 2])?,
            kernel_format: KernelFormat::HWIO,
            group: 2,
            data,
            kernel,
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 1),
        },
    );

    let data = ArrayD::zeros(vec![2, 1]);
    let kernel = ArrayD::zeros(vec![1, 1, 1]);
    suite.add(
        "same_upper",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2]).unwrap(),
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data,
            kernel,
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1),
        },
    );
    suite.add(
        "explicit_dnn_left",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::arr2(&[[0.0]]).into_dyn(),
            kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(2), tvec!(0)),
            strides: tvec!(1),
        },
    );

    suite.add(
        "explicit_dnn_right_0",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::arr2(&[[0.0]]).into_dyn(),
            kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(0), tvec!(2)),
            strides: tvec!(1),
        },
    );

    suite.add(
        "explicit_dnn_right_1",
        ConvProblem {
            shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: tract_ndarray::arr2(&[[0.0]]).into_dyn(),
            kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Explicit(tvec!(0), tvec!(1)),
            strides: tvec!(1),
        },
    );

    suite.add(
        "dnn_2d",
        ConvProblem {
            shape_in: DataFormat::NCHW.from_n_c_hw(1, 2, [1, 1]).unwrap(),
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr4(&[[[[0.0]], [[1.0]]]]).into_dyn(),
            kernel: arr4(&[[[[0.0]], [[1.0]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1),
        },
    );

    suite.add(
        "group_owhi_0",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [1]).unwrap(),
            kernel_format: KernelFormat::OHWI,
            group: 2,
            data: arr2(&[[0.0], [0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0, 0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "dw_k4_0",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [8]).unwrap(),
            kernel_format: KernelFormat::OHWI,
            group: 2,
            data: arr2(&[
                [0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0],
            ])
            .into_dyn(),
            kernel: arr3(&[[[1.0f32, 2.0], [-1.0, 1.0], [0.0, 3.0], [3.0, 0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "dw_k4_d2_0",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [6, 2]).unwrap(),
            kernel_format: KernelFormat::HWIO,
            group: 2,
            data: arr3(&[
                [[0.0f32, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            ])
            .into_dyn(),
            kernel: arr4(&[[[[0.0f32], [0.0]], [[0.0], [0.0]]], [[[0.0], [0.0]], [[0.0], [-1.0]]]])
                .into_dyn(),
            bias: None,
            pad: PaddingSpec::SameUpper,
            strides: tvec!(1, 1),
        },
    );

    suite.add(
        "from_1d_to_2d",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [2]).unwrap(),
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr2(&[[0.0f32, 0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0f32, 0.0]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "trivial_nchw",
        ConvProblem {
            shape_in: DataFormat::NCHW.from_n_c_hw(1, 1, [1, 1]).unwrap(),
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr4(&[[[[0.0f32]]]]).into_dyn(),
            kernel: arr4(&[[[[0.0f32]]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1),
        },
    );

    let mut data = Tensor::zero::<f32>(&[1, 5, 6]).unwrap();
    *data.as_slice_mut::<f32>().unwrap().last_mut().unwrap() = 1.0;
    let mut kernel = Tensor::zero::<f32>(&[1, 1, 3, 2]).unwrap();
    *kernel.as_slice_mut::<f32>().unwrap().last_mut().unwrap() = 1.0;
    suite.add(
        "pack_0",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [5, 6]).unwrap(),
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: data.into_array::<f32>().unwrap(),
            kernel: kernel.into_array::<f32>().unwrap(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1),
        },
    );

    suite.add(
        "pack_1",
        ConvProblem {
            shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [5]).unwrap(),
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr2(&[[0.0f32, -2.0, 0.0, 0.0, 0.0]]).into_dyn(),
            kernel: arr3(&[[[1f32]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    suite.add(
        "bug_metal_0",
        ConvProblem {
            shape_in: DataFormat::NHWC.from_n_c_hw(2, 1, [4]).unwrap(),
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: arr3(&[[[0f32], [0.], [0.], [0.]], [[0.], [0.], [0.], [1.]]]).into_dyn(),
            kernel: arr3(&[[[0f32]], [[1.]]]).into_dyn(),
            bias: None,
            pad: PaddingSpec::Valid,
            strides: tvec!(1),
        },
    );

    Ok(suite)
}
