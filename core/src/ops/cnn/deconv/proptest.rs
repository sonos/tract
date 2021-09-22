use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::*;
use crate::ops::nn::*;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_ndarray::{prelude::*, *};
use DataFormat::*;
use KernelFormat::*;

#[derive(Debug)]
struct DeconvProblem {
    optimized: bool,
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

fn tensor(shape: &[usize]) -> BoxedStrategy<ArrayD<f32>> {
    let shape = shape.to_vec();
    let len = shape.iter().product::<usize>();
    vec(any::<i8>().prop_map(|i| i as f32), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(&*shape, vec).unwrap())
        .boxed()
}

impl Arbitrary for DeconvProblem {
    type Strategy = BoxedStrategy<DeconvProblem>;
    type Parameters = ();
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (any::<bool>(), 1usize..4)
            .prop_flat_map(|(opt, georank)| {
                (
                    Just(opt),
                    any::<DataFormat>(),
                    any::<KernelFormat>(),
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
                |(_, _, _, pad, _, _, _, hwk, _, strides, dilations, _)| {
                    pad == &PaddingSpec::Valid
                        || tract_itertools::izip!(hwk, dilations, strides)
                            .all(|(k, d, s)| (k - 1) * d > s - 1)
                },
            )
            .prop_flat_map(
                |(
                    opt,
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
                            kernel_shape.insert(0, co_over_group);
                            kernel_shape.insert(1, ci_over_group * group);
                        }
                        HWIO => {
                            kernel_shape.push(ci_over_group * group);
                            kernel_shape.push(co_over_group);
                        }
                    };
                    let data_shape = df.from_n_c_hw(n, ci_over_group * group, &hwi).unwrap();
                    (
                        Just(opt),
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
                    optimized,
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
                        optimized,
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
    fn as_op(&self) -> DeconvUnary {
        let pool_spec = PoolSpec::new(
            self.data_format,
            self.kernel_format.spatial_shape(self.kernel.shape()).into(),
            self.padding.clone(),
            Some(self.dilations.clone()),
            Some(self.strides.clone()),
            Some(match self.kernel_format {
                KernelFormat::OIHW => self.kernel.shape()[0] * self.group,
                KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1] * self.group,
            }),
        );
        let op = DeconvUnary::new(
            pool_spec,
            self.kernel_format,
            self.kernel.clone().into_arc_tensor(),
            self.bias.as_ref().map(|b| b.clone().into_arc_tensor()),
            self.adjustments.clone(),
            self.group,
        );
        let fact = TypedFact::shape::<f32, _>(self.input.shape());
        op.output_facts(&[&fact]).unwrap();
        op
    }

    fn model_eval(&self) -> ArrayD<f32> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", TypedFact::shape::<f32, _>(self.input.shape())).unwrap();
        let output = model.wire_node("deconv", self.as_op(), &[src]).unwrap();
        model.set_output_outlets(&output).unwrap();
        let model = model.into_optimized().unwrap();
        let mut outputs =
            model.into_runnable().unwrap().run(tvec!(self.input.clone().into_tensor())).unwrap();
        outputs.remove(0).into_tensor().into_array().unwrap().into_dimensionality().unwrap()
    }

    fn op_eval(&self) -> ArrayD<f32> {
        let op = self.as_op();
        let mut outputs = op.eval(tvec!(self.input.clone().into_arc_tensor())).unwrap();
        outputs.remove(0).into_tensor().into_array().unwrap().into_dimensionality().unwrap()
    }

    fn reference(&self) -> ArrayD<f32> {
        use std::iter::once;
        let co = match self.kernel_format {
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1] * self.group,
            KernelFormat::OIHW => self.kernel.shape()[0] * self.group,
        };
        let input_shape = self.data_format.shape(self.input.shape()).unwrap();
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
            valid_output_shape_geo.clone()
        } else {
            tract_itertools::izip!(input_shape.hw_dims(), &self.strides)
                .map(|(i, s)| i * s)
                .collect()
        };
        let output_shape = self.data_format.from_n_c_hw(n, co, output_shape_geo).unwrap();
        let mut output = ArrayD::zeros(&*output_shape.shape);
        if let Some(b) = &self.bias {
            let mut bias_shape = tvec!(1; output_shape.rank());
            bias_shape[output_shape.c_axis()] = co;
            let b = b.clone().into_shape(&*bias_shape).unwrap();
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
                                let i = self
                                    .data_format
                                    .from_n_c_hw(n, ci + g * ci_per_group, hwi.slice())
                                    .unwrap();
                                let o = self
                                    .data_format
                                    .from_n_c_hw(n, co + g * co_per_group, hwo)
                                    .unwrap();
                                let k: TVec<usize> = match self.kernel_format {
                                    OIHW => once(co)
                                        .chain(once(ci + ci_per_group * g))
                                        .chain(hwk.slice().iter().cloned())
                                        .collect(),
                                    HWIO => hwk
                                        .slice()
                                        .iter()
                                        .cloned()
                                        .chain(once(ci + ci_per_group * g))
                                        .chain(once(co))
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
        output
    }

    fn check(&self) -> Result<(), TestCaseError> {
        if self.optimized {
            prop_assert_eq!(self.model_eval(), self.reference());
        } else {
            prop_assert_eq!(self.op_eval(), self.reference());
        }
        Ok(())
    }
}

proptest::proptest! {
    #[test]
    fn prop(pb in any::<DeconvProblem>()) {
        pb.check().unwrap();
    }
}

#[test]
fn test_trivial_0() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_hwc_0() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_geo_0() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_hwio_0() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_strides_1() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_same_upper_1() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_same_upper_dil() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_same_upper_strides() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_channel_0() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_group_0() {
    let pb = DeconvProblem {
        optimized: false,
        data_format: HWC,
        kernel_format: OIHW,
        padding: PaddingSpec::Valid,
        input: arr2(&[[0.0, 0.0]]).into_dyn(),
        kernel: arr3(&[[[0.0], [0.0]]]).into_dyn(),
        bias: None,
        strides: tvec!(1),
        dilations: tvec!(1),
        adjustments: tvec!(0),
        group: 2,
    };
    pb.check().unwrap();
}

#[test]
fn test_group_1() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_group_2() {
    let pb = DeconvProblem {
        optimized: false,
        data_format: HWC,
        kernel_format: OIHW,
        padding: PaddingSpec::Valid,
        input: ndarray::arr2(&[[1.0, 0.0]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1.0], [0.0]]]).into_dyn(),
        bias: None,
        strides: tvec!(1),
        dilations: tvec!(1),
        adjustments: tvec!(0),
        group: 2,
    };
    pb.check().unwrap();
}

#[test]
fn test_group_3() {
    let pb = DeconvProblem {
        optimized: false,
        data_format: HWC,
        kernel_format: OIHW,
        padding: PaddingSpec::Valid,
        input: ndarray::arr2(&[[0.0, 1.0]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0], [0.0]], [[1.0], [0.0]]]).into_dyn(),
        bias: None,
        strides: tvec!(1),
        dilations: tvec!(1),
        adjustments: tvec!(0),
        group: 2,
    };
    pb.check().unwrap();
}

#[test]
fn test_bias_0() {
    let pb = DeconvProblem {
        optimized: false,
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
    };
    pb.check().unwrap();
}

#[test]
fn test_rank_5_with_group() {
    let pb = DeconvProblem {
        optimized: false,
        data_format: HWC,
        kernel_format: OIHW,
        padding: PaddingSpec::Valid,
        input: arr4(&[[[[0.0, 0.0, 0.0, 1.0]]]]).into_dyn(),
        kernel: arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .into_shape(vec![1, 4, 1, 2, 1])
            .unwrap()
            .into_dyn(),
        bias: None,
        strides: tvec!(1, 1, 1),
        dilations: tvec!(1, 1, 1),
        adjustments: tvec!(0, 0, 0),
        group: 2,
    };
    pb.check().unwrap();
}

#[test]
fn test_issue_512_simplified() {
    let pb = DeconvProblem {
        optimized: false,
        data_format: NCHW,
        kernel_format: OIHW,
        padding: PaddingSpec::Valid,
        input: ndarray::Array4::zeros([1, 4, 1, 1]).into_dyn(),
        kernel: ndarray::Array4::zeros([1, 4, 1, 1]).into_dyn(),
        bias: None,
        strides: tvec!(1, 1),
        dilations: tvec!(1, 1),
        adjustments: tvec!(0, 0),
        group: 2,
    };
    pb.check().unwrap();
}
