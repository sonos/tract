use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::ops::cnn::conv::KernelFormat;
use tract_core::ops::cnn::*;
use tract_core::ops::nn::*;
use tract_ndarray as ndarray;
use tract_ndarray::{prelude::*, *};
use DataFormat::*;
use KernelFormat::*;

use crate::data_format;
use crate::kernel_format;
use crate::tensor;

#[derive(Debug, Clone, Default)]
pub struct DeconvProblemParams {}

#[derive(Debug, Clone)]
struct DeconvProblem {
    data_format: DataFormat,
    kernel_format: KernelFormat,
    padding: PaddingSpec,
    input: ArrayD<f32>,
    kernel: ArrayD<f32>,
    bias: Option<ArrayD<f32>>,
    strides: TVec<usize>,
    dilations: TVec<usize>,
    adjustments: TVec<usize>,
    group: usize,
}

impl Arbitrary for DeconvProblem {
    type Strategy = BoxedStrategy<DeconvProblem>;
    type Parameters = DeconvProblemParams;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (1usize..4)
            .prop_flat_map(|georank| {
                (
                    data_format(),
                    kernel_format(),
                    prop_oneof![Just(PaddingSpec::Valid), Just(PaddingSpec::SameUpper)],
                    1usize..3,                         // n
                    1usize..4,                         // ci / group
                    1usize..4,                         // co / group
                    vec(1usize..4, georank..=georank), // kernel shape
                    vec(1usize..8, georank..=georank), // image shape
                    vec(1usize..4, georank..=georank), // strides
                    vec(1usize..4, georank..=georank), // dilations
                    1usize..4,                         // group
                )
            })
            .prop_filter(
                "dilation, strides and shapes in SAME",
                |(_, _, pad, _, _, _, hwk, _, strides, dilations, _)| {
                    pad == &PaddingSpec::Valid
                        || tract_itertools::izip!(hwk, dilations, strides)
                            .all(|(k, d, s)| (k - 1) * d > s - 1)
                },
            )
            .prop_flat_map(
                |(
                    df,
                    kf,
                    pad,
                    n,
                    ci_over_group,
                    co_over_group,
                    hwk,
                    hwi,
                    strides,
                    dilations,
                    group,
                )| {
                    let mut kernel_shape = hwk;
                    match kf {
                        OIHW => {
                            kernel_shape.insert(0, co_over_group * group);
                            kernel_shape.insert(1, ci_over_group);
                        }
                        HWIO => {
                            kernel_shape.push(ci_over_group * group);
                            kernel_shape.push(co_over_group);
                        }
                        OHWI => {
                            kernel_shape.insert(0, co_over_group);
                            kernel_shape.push(ci_over_group * group);
                        }
                    };
                    let data_shape = df.from_n_c_hw(n, ci_over_group * group, hwi).unwrap();
                    (
                        Just(df),
                        Just(kf),
                        Just(pad),
                        tensor(&data_shape.shape),
                        tensor(&kernel_shape),
                        proptest::option::of(tensor(&[co_over_group * group])),
                        Just(strides),
                        Just(dilations),
                        Just(group),
                    )
                },
            )
            .prop_map(
                |(
                    data_format,
                    kernel_format,
                    padding,
                    input,
                    kernel,
                    bias,
                    strides,
                    dilations,
                    group,
                )| {
                    let adjustments = tvec!(0; kernel.ndim() - 2); // FIXME maybe
                    DeconvProblem {
                        data_format,
                        kernel_format,
                        padding,
                        input,
                        kernel,
                        bias,
                        strides: strides.into(),
                        dilations: dilations.into(),
                        adjustments,
                        group,
                    }
                },
            )
            .boxed()
    }
}

impl DeconvProblem {
    fn as_op(&self) -> TractResult<Deconv> {
        let pool_spec = PoolSpec::new(
            self.data_format,
            self.kernel_format.spatial_shape(self.kernel.shape()).into(),
            self.padding.clone(),
            Some(self.dilations.clone()),
            Some(self.strides.clone()),
            self.kernel_format.input_channels(self.kernel.shape(), self.group).into_owned(),
            self.kernel_format.output_channels(self.kernel.shape(), self.group).into_owned(),
        );
        let op = Deconv::new(pool_spec, self.kernel_format, self.adjustments.clone(), self.group);
        Ok(op)
    }

    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact(self.input.shape()))?;
        let kernel = model.add_const("kernel", self.kernel.clone().into_tensor())?;
        let bias =
            self.bias.as_ref().map(|b| b.clone().into_tensor()).unwrap_or_else(|| tensor0(0f32));
        let bias = model.add_const("bias", bias)?;
        let output = model.wire_node(
            "deconv",
            self.as_op().context("Generating op")?,
            &[src, kernel, bias],
        )?;
        model.set_output_outlets(&output)?;
        Ok(model)
    }

    fn reference(&self) -> TractResult<ArrayD<f32>> {
        use std::iter::once;
        let co = match self.kernel_format {
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1] * self.group,
            KernelFormat::OIHW => self.kernel.shape()[0],
            KernelFormat::OHWI => self.kernel.shape()[0] * self.group,
        };
        let input_shape = self.data_format.shape(self.input.shape())?;
        let n = if self.data_format.has_n() { self.input.shape()[0] } else { 1 };
        let kernel_hwdims = self.kernel_format.spatial_shape(self.kernel.shape());
        let valid_output_shape_geo: TVec<usize> = tract_itertools::izip!(
            input_shape.hw_dims(),
            kernel_hwdims,
            self.strides.iter(),
            self.dilations.iter()
        )
        .map(|(i, k, s, d)| (i - 1) * s + (k - 1) * d + 1)
        .collect();
        let paddings: TVec<(usize, usize)> = if self.padding == PaddingSpec::Valid {
            tvec![(0, 0); valid_output_shape_geo.len()]
        } else {
            tract_itertools::izip!(input_shape.hw_dims(), &valid_output_shape_geo, &self.strides)
                .map(|(i, o, s)| o - i * s)
                .map(|total| (total / 2, total - total / 2))
                .collect()
        };
        let output_shape_geo = if self.padding == PaddingSpec::Valid {
            valid_output_shape_geo
        } else {
            tract_itertools::izip!(input_shape.hw_dims(), &self.strides)
                .map(|(i, s)| i * s)
                .collect()
        };
        let output_shape = self.data_format.from_n_c_hw(n, co, output_shape_geo)?;
        let mut output = ArrayD::zeros(&*output_shape.shape);
        if let Some(b) = &self.bias {
            let mut bias_shape = tvec!(1; output_shape.rank());
            bias_shape[output_shape.c_axis()] = co;
            let b = b.clone().into_shape_with_order(&*bias_shape)?;
            output += &b;
        }
        let co_per_group = co / self.group;
        let ci_per_group = input_shape.c() / self.group;
        for n in 0..n {
            for g in 0..self.group {
                for co in 0..co_per_group {
                    for ci in 0..ci_per_group {
                        for hwi in indices(input_shape.hw_dims()) {
                            for hwk in indices(kernel_hwdims) {
                                let hwo: TVec<isize> = tract_itertools::izip!(
                                    hwi.slice().iter(),
                                    hwk.slice().iter(),
                                    self.strides.iter(),
                                    self.dilations.iter(),
                                    paddings.iter(),
                                )
                                .map(|(i, k, s, d, p)| (i * s + k * d) as isize - p.0 as isize)
                                .collect();
                                let hwo: TVec<usize> = if hwo.iter().all(|x| *x >= 0) {
                                    hwo.iter().map(|x| *x as usize).collect()
                                } else {
                                    continue;
                                };
                                let i = self.data_format.from_n_c_hw(
                                    n,
                                    ci + g * ci_per_group,
                                    hwi.slice(),
                                )?;
                                let o =
                                    self.data_format.from_n_c_hw(n, co + g * co_per_group, hwo)?;
                                let k: TVec<usize> = match self.kernel_format {
                                    OIHW => once(co + co_per_group * g)
                                        .chain(once(ci))
                                        .chain(hwk.slice().iter().cloned())
                                        .collect(),
                                    HWIO => hwk
                                        .slice()
                                        .iter()
                                        .cloned()
                                        .chain(once(ci + ci_per_group * g))
                                        .chain(once(co))
                                        .collect(),
                                    OHWI => once(co)
                                        .chain(hwk.slice().iter().cloned())
                                        .chain(once(ci + ci_per_group * g))
                                        .collect(),
                                };
                                if let Some(cell) = output.get_mut(&*o.shape) {
                                    *cell += self.input[&*i.shape] * self.kernel[&*k]
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(output)
    }
}

impl Test for DeconvProblem {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().context("Running reference")?.into_tensor();
        let mut model = self.tract().context("Generating model")?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut output = runtime.prepare(model)?.run(tvec![self.input.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<DeconvProblem>("proptest", DeconvProblemParams::default());

    suite.add(
        "trivial_0",
        DeconvProblem {
            data_format: NCHW,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr4(&[[[[0.0]]]]).into_dyn(),
            kernel: arr4(&[[[[0.0]]]]).into_dyn(),
            bias: None,
            strides: tvec!(1, 1),
            dilations: tvec!(1, 1),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "hwc_0",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr3(&[[[0.0]], [[0.0]]]).into_dyn(),
            kernel: arr4(&[[[[0.0]]]]).into_dyn(),
            bias: None,
            strides: tvec!(1, 1),
            dilations: tvec!(1, 1),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "geo_0",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr3(&[[[0.0]]]).into_dyn(),
            kernel: arr4(&[[[[0.0], [0.0]]]]).into_dyn(),
            bias: None,
            strides: tvec!(1, 1),
            dilations: tvec!(1, 1),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "hwio_0",
        DeconvProblem {
            data_format: HWC,
            kernel_format: HWIO,
            padding: PaddingSpec::Valid,
            input: arr3(&[[[0.0]]]).into_dyn(),
            kernel: arr4(&[[[[0.0, 0.0]]]]).into_dyn(),
            bias: None,
            strides: tvec!(1, 1),
            dilations: tvec!(1, 1),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "strides_1",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0.0], [1.0]]).into_dyn(),
            kernel: arr3(&[[[1.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(2),
            dilations: tvec!(1),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "same_upper_1",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::SameUpper,
            input: arr2(&[[0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0, 0.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "same_upper_dil",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::SameUpper,
            input: arr2(&[[0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0, 0.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(2),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "same_upper_strides",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::SameUpper,
            input: arr2(&[[0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0, 0.0, 0.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(2),
            dilations: tvec!(1),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "channel_0",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0]], [[0.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 1,
        },
    );

    suite.add(
        "group_0",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0.0, 0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0]], [[0.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 2,
        },
    );

    suite.add(
        "group_1",
        DeconvProblem {
            data_format: HWC,
            kernel_format: HWIO,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0.0, 0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0], [0.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 2,
        },
    );

    suite.add(
        "group_2",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: ndarray::arr2(&[[1.0, 0.0]]).into_dyn(),
            kernel: ndarray::arr3(&[[[1.0]], [[0.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 2,
        },
    );

    suite.add(
        "group_3",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: ndarray::arr2(&[[0.0, 1.0]]).into_dyn(),
            kernel: ndarray::arr3(&[[[0.0]], [[1.0]], [[0.0]], [[0.0]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 2,
        },
    );

    suite.add(
        "group_4",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0f32, 1.]]).into_dyn(),
            kernel: arr3(&[[[0f32]], [[1.]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 2,
        },
    );

    suite.add(
        "group_hwio_0",
        DeconvProblem {
            data_format: CHW,
            kernel_format: HWIO,
            padding: PaddingSpec::Valid,
            input: Array2::from_shape_vec((4, 1), vec![0f32, 0., 1., 0.]).unwrap().into_dyn(),
            kernel: Array3::from_shape_vec((1, 4, 1), vec![0f32, 0., 1., 0.]).unwrap().into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 2,
        },
    );

    suite.add(
        "bias_0",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0]]]).into_dyn(),
            bias: Some(arr1(&[1.0f32]).into_dyn()),
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 1,
        },
    );

    suite.add(
        "bias_1",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0.0], [0.0]]).into_dyn(),
            kernel: arr3(&[[[0.0]]]).into_dyn(),
            bias: Some(arr1(&[1.0f32]).into_dyn()),
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 1,
        },
    );

    suite.add(
        "bias_2",
        DeconvProblem {
            data_format: CHW,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0f32, 1.]]).into_dyn(),
            kernel: arr3(&[[[1f32]], [[0.]]]).into_dyn(),
            bias: Some(arr1(&[0f32, 0.]).into_dyn()),
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 1,
        },
    );

    suite.add(
        "bias_group_0",
        DeconvProblem {
            data_format: CHW,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr2(&[[0f32], [1.]]).into_dyn(),
            kernel: arr3(&[[[1f32]], [[0.]]]).into_dyn(),
            bias: Some(arr1(&[0f32, 0.]).into_dyn()),
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 2,
        },
    );

    suite.add(
        "rank_5_with_group",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr4(&[[[[0.0, 0.0, 0.0, 1.0]]]]).into_dyn(),
            kernel: arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .into_shape_with_order(vec![2, 2, 1, 2, 1])
                .unwrap()
                .into_dyn(),
            bias: None,
            strides: tvec!(1, 1, 1),
            dilations: tvec!(1, 1, 1),
            adjustments: tvec!(0, 0, 0),
            group: 2,
        },
    );

    suite.add(
        "issue_512_simplified",
        DeconvProblem {
            data_format: NCHW,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: ndarray::Array4::zeros([1, 4, 1, 1]).into_dyn(),
            kernel: ndarray::Array4::zeros([2, 2, 1, 1]).into_dyn(),
            bias: None,
            strides: tvec!(1, 1),
            dilations: tvec!(1, 1),
            adjustments: tvec!(0, 0),
            group: 2,
        },
    );

    suite.add(
        "issue_optim_2d",
        DeconvProblem {
            data_format: HWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: ndarray::Array3::zeros([2, 2, 1]).into_dyn(),
            kernel: ndarray::Array4::zeros([1, 1, 1, 1]).into_dyn(),
            bias: None,
            strides: tvec!(1, 2),
            dilations: tvec!(1, 1),
            adjustments: tvec!(0, 0),
            group: 1,
        },
    );

    suite.add(
        "foo",
        DeconvProblem {
            data_format: NHWC,
            kernel_format: OIHW,
            padding: PaddingSpec::Valid,
            input: arr3(&[[[0f32]], [[1.]]]).into_dyn(),
            kernel: arr3(&[[[1f32]]]).into_dyn(),
            bias: None,
            strides: tvec!(1),
            dilations: tvec!(1),
            adjustments: tvec!(0),
            group: 1,
        },
    );

    Ok(suite)
}
