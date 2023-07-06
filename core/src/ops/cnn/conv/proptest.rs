use super::KernelFormat;
use crate::ops::cnn::*;
use crate::ops::nn::*;
use crate::setup_test_logger;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_itertools::izip;
use tract_ndarray::arr3;
use tract_ndarray::prelude::*;

#[derive(Debug)]
struct ConvProblem {
    shape_in: DataShape,
    kernel_format: KernelFormat,
    group: usize,
    data: ArrayD<f32>,
    kernel: ArrayD<f32>,
    bias: Option<ArrayD<f32>>,
    pad: PaddingSpec,
    strides: TVec<usize>,
}

impl ConvProblem {
    fn geo_ker(&self) -> &[usize] {
        &self.kernel.shape()[self.kernel_format.h_axis()..][..self.shape_in.hw_rank()]
    }

    fn reference(&self) -> ArrayD<f32> {
        setup_test_logger();
        assert_eq!(self.data.shape(), &*self.shape_in.shape, "inconsistent shapes in test");
        let n = *self.shape_in.n().unwrap_or(&1);
        let ci_per_g = self.shape_in.c() / self.group;
        let co_per_g = match self.kernel_format {
            KernelFormat::OIHW => self.kernel.shape()[0] / self.group,
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1],
            KernelFormat::OHWI => self.kernel.shape()[0],
        };
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
            PaddingSpec::Explicit(l, r, ceil) => {
                izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides, l, r)
                    .map(|(input, k, stride, l, r)| {
                        let dil = 1;
                        let kf = (k - 1) * dil + 1;
                        let out = if *ceil {
                            (input + l + r).saturating_sub(kf).divceil(*stride) + 1
                        } else {
                            (input + l + r).saturating_sub(kf) / *stride + 1
                        };
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
                for geo_out in tract_ndarray::indices(shape_out.hw_dims()) {
                    let mut output_coords: TVec<usize> = geo_out.slice().into();
                    if self.shape_in.fmt.has_n() {
                        output_coords.insert(0, n);
                    }
                    output_coords.insert(shape_out.c_axis(), 0);
                    for geo_ker in tract_ndarray::indices(self.geo_ker()) {
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
            out += &bias.clone().into_shape(shape).unwrap();
        }
        out
    }

    fn tract(&self) -> anyhow::Result<ArrayD<f32>> {
        setup_test_logger();
        assert_eq!(self.data.shape(), &*self.shape_in.shape, "inconsistent shapes in test");
        let mut model = TypedModel::default();
        let wire = model.add_source("input", f32::fact(&self.shape_in.shape))?;
        let co = match self.kernel_format {
            KernelFormat::OIHW => self.kernel.shape()[0],
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1] * self.group,
            KernelFormat::OHWI => self.kernel.shape()[0] * self.group,
        };
        let op = ConvUnary::new(
            PoolSpec::new(
                self.shape_in.fmt,
                self.geo_ker().into(),
                self.pad.clone(),
                None,
                Some(self.strides.clone()),
                Some(co),
            ),
            self.kernel_format,
            self.kernel.clone().into_arc_tensor(),
            self.group,
            self.bias.clone().map(|a| a.into_arc_tensor()),
            None,
        );
        let wire = model.wire_node("conv", op, &[wire])?[0];
        model.set_output_outlets(&[wire])?;
        dbg!(&model);
        let mut output =
            model.into_optimized()?.into_runnable()?.run(tvec![self.data.clone().into_tvalue()])?;
        output.remove(0).into_tensor().into_array::<f32>()
    }
}

fn explicit_padding(rank: usize) -> impl Strategy<Value = PaddingSpec> {
    let left = vec(0..3usize, rank..rank + 1);
    let right = vec(0..3usize, rank..rank + 1);
    (left, right, any::<bool>()).prop_map(|(l, r, b)| PaddingSpec::Explicit(l.into(), r.into(), b))
}

impl Arbitrary for ConvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<ConvProblem>;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            any::<DataFormat>(),
            any::<KernelFormat>(),
            1usize..=3,
            1usize..=4,
            1usize..=4,
            1usize..=3,
            (1usize..=3).prop_flat_map(shapes),
        )
            .prop_flat_map(|(df, kf, n, mut ci0, co0, group, (mut ker_shape, data_shape))| {
                // FIXME in HWIO order, only regular and depthwise are supported
                if kf == KernelFormat::HWIO && group > 1 {
                    ci0 = 1;
                }
                let shape_in = df.from_n_c_hw(n, ci0 * group, data_shape).unwrap();
                let data_in = tensor(shape_in.shape.iter().cloned().collect());
                let pad = prop_oneof![
                    Just(PaddingSpec::Valid).boxed(),
                    Just(PaddingSpec::SameUpper).boxed(),
                    Just(PaddingSpec::SameLower).boxed(),
                    explicit_padding(ker_shape.len()).boxed()
                ];
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
                let kernel = tensor(ker_shape);
                let bias = proptest::option::of(tensor(vec![co0 * group]));
                (Just((kf, shape_in, group)), pad, data_in, kernel, bias, strides)
            })
            .prop_map(|((kernel_format, shape_in, group), pad, data, kernel, bias, strides)| {
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

impl Arbitrary for KernelFormat {
    type Parameters = ();
    type Strategy = BoxedStrategy<KernelFormat>;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        prop_oneof!(Just(KernelFormat::OIHW), Just(KernelFormat::HWIO)).boxed()
    }
}

impl Arbitrary for DataFormat {
    type Parameters = ();
    type Strategy = BoxedStrategy<DataFormat>;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        prop_oneof!(
            Just(DataFormat::HWC),
            Just(DataFormat::CHW),
            Just(DataFormat::NHWC),
            Just(DataFormat::NCHW)
        )
        .boxed()
    }
}

pub fn tensor(shape: Vec<usize>) -> BoxedStrategy<ArrayD<f32>> {
    let len = shape.iter().product::<usize>();
    vec((-10i8..=10i8).prop_map(|i| i as f32), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap())
        .boxed()
}

pub fn shapes(rank: usize) -> BoxedStrategy<(Vec<usize>, Vec<usize>)> {
    vec((1usize..4, 0usize..5).prop_map(|(k, exceed)| (k, k + exceed)), rank..=rank)
        .prop_map(|v| v.into_iter().unzip())
        .boxed()
}

proptest::proptest! {
    #[test]
    fn prop(pb in any::<ConvProblem>()) {
        pb.tract().unwrap().into_tensor().close_enough(&pb.reference().into_tensor(), true).unwrap();
    }
}

#[test]
fn trivial_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[0.0f32]]]).into_dyn(),
        kernel: arr4(&[[[[0.0f32]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn trivial_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 1, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn trivial_2() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[1.0f32], [0.0]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn trivial_3() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32], [1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn nchw_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[0f32, 1.0]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1f32]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

/*
#[test]
fn group_2() -> anyhow::Result<()> {
let pb = ConvProblem {
shape_in: DataFormat::HWC.from_n_c_hw(1, 4, &[1])?,
shape_out: DataFormat::HWC.from_n_c_hw(1, 2, &[1])?,
kernel_format: KernelFormat::HWIO,
group: 2,
data: ndarray::arr2(&[[0.0f32, 0.0, 1.0, 0.0]]).into_dyn(),
kernel: ndarray::arr3(&[[[0.0f32], [0.0], [1.0], [0.0]]]).into_dyn(),
bias: None,
};
assert_eq!(pb.tract()?, pb.reference());
Ok(())
}
*/

#[test]
fn group_3() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_4() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32]], [[0.0]], [[0.0]], [[1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_5() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1, 1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
        kernel: tensor4(&[[[[0.0f32]]], [[[0.0]]], [[[0.0]]], [[[0.0]]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_6() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
        kernel: tensor3(&[[[0.0f32]], [[0.0]], [[0.0]], [[0.0]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_7() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(1, 2, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr3(&[[[0.0f32, 0.0], [0.0, 1.0]]]).into_dyn(),
        kernel: tensor3(&[[[0.0f32, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 1.0]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_8() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 4, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 0.0, 0.0, 1.0]]).into_dyn(),
        kernel: tensor3(&[[[0.0f32], [0.0]], [[0.0], [0.0]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_9() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 0.0], [0.0, 1.0]]).into_dyn(),
        kernel: tensor3(&[[[0.0f32]], [[0.0]], [[0.0]], [[1.0]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_10() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [2, 1, 4])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::ArrayD::<f32>::zeros(vec![2, 2, 1, 4]),
        kernel: ndarray::ArrayD::from_elem(vec![4, 1, 1, 1, 2], 1.0f32),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_11() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: tract_ndarray::arr2(&[[0.0, 1.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[
            [[0.0]],
            [[0.0]],
            [[0.0]],
            [[0.0]],
            [[0.0]],
            [[0.0]],
            [[0.0]],
            [[1.0]],
        ])
        .into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_12() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::HWIO,
        group: 2,
        data: tract_ndarray::arr2(&[[0.0, 0.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0], [0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_13() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::HWIO,
        group: 2,
        data: tract_ndarray::arr2(&[[0.0, 1.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0, 0.0], [1.0, 0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_bias_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![4, 1, 1]),
        bias: Some(ndarray::ArrayD::<f32>::zeros(vec![4])),
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_bias_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: tract_ndarray::arr2(&[[0.0, 0.0]]).into_dyn(),
        kernel: tract_ndarray::ArrayD::<f32>::zeros(vec![4, 1, 1]),
        bias: Some(tract_ndarray::arr1(&[0.0, 0.0, 0.0, 1.0]).into_dyn()),
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn bias_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::ArrayD::<f32>::zeros(vec![2, 1]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2]),
        bias: Some(ndarray::ArrayD::<f32>::zeros(vec![1])),
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn bias_1() -> anyhow::Result<()> {
    let kernel = tract_ndarray::ArrayD::<f32>::zeros(vec![2, 1, 2]);
    let data = tract_ndarray::ArrayD::<f32>::zeros(vec![2, 1]);
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data,
        kernel,
        bias: Some(arr1(&[0.0f32, 1.0]).into_dyn()),
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn bias_chw_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr2(&[[0f32, 0., 0.]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0f32]], [[0.]], [[0.]]]).into_dyn(),
        bias: Some(ndarray::arr1(&[0f32, 0., 1.]).into_dyn()),
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn batch_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(2, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::ArrayD::<f32>::zeros(vec![2, 2, 1]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2]),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn batch_1() -> anyhow::Result<()> {
    let data = arr3(&[[[0.0f32]], [[1.0]]]).into_dyn();
    let kernel = arr3(&[[[1.0f32]]]).into_dyn();
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(2, 1, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data,
        kernel,
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn batch_2() -> anyhow::Result<()> {
    let data = arr3(&[[[0.0f32, 0.0]], [[0.0, 0.0]]]).into_dyn();
    let kernel = arr3(&[[[0.0f32]]]).into_dyn();
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(2, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data,
        kernel,
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn bias_3d_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [1, 1, 2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 1, 2]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 1, 1, 1]),
        bias: Some(ndarray::ArrayD::<f32>::ones(vec![1])),
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn batch_3d() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(1, 1, [2, 2, 1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2, 2, 1]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 1, 1, 1]),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::ArrayD::<f32>::zeros(vec![1, 1]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 1]),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0], [1.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0, 1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_2() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr3(&[[[0.0], [0.0]], [[0.0], [1.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0], [0.0]]], [[[0.0], [1.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_2d_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr3(&[[[0.0], [0.0], [1.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_2d_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr3(&[[[0.0], [0.0]], [[1.0], [0.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0, 1.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0], [0.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0], [0.0], [1.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2_dnn_padding_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [6])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Explicit(tvec!(1), tvec!(1), false),
        strides: tvec!(2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2_dnn_padding_2() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Explicit(tvec!(1), tvec!(1), false),
        strides: tvec!(2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2_dnn_padding_3() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0],[0.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Explicit(tvec!(0), tvec!(0), true),
        strides: tvec!(2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0], [0.0], [1.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0, 0.0, 1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(3),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2d_same_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr3(&[[[0.0], [0.0], [1.0]]]).into_dyn(),
        kernel: arr4(&[[[[1.0, 0.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2d_same_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::ArrayD::<f32>::zeros(vec![2, 3, 1]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 1, 3]),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2d_same_2() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr3(&[[[0.0], [0.0], [1.0]], [[0.0], [0.0], [0.0]]]).into_dyn(),
        kernel: arr4(&[[[[1.0, 0.0, 0.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_two_axes() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(2, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn lazy_im2col_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0, 0.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0, 0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn lazy_im2col_big() -> anyhow::Result<()> {
    let mut kernel = tract_ndarray::ArrayD::<f32>::zeros(vec![1, 4, 1, 3, 2]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 4, [2, 5, 4])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::ArrayD::<f32>::zeros(vec![4, 2, 5, 4]),
        kernel,
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(2, 2, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn lazy_im2col_big_2() -> anyhow::Result<()> {
    let mut kernel = tract_ndarray::ArrayD::<f32>::zeros(vec![1, 4, 1, 3, 2]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 4, [2, 5, 4])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::ArrayD::<f32>::zeros(vec![1, 2, 5, 4, 4]),
        kernel,
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(2, 3, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn depthwise_0() -> anyhow::Result<()> {
    let mut kernel = tract_ndarray::ArrayD::<f32>::zeros(vec![2, 2, 2, 1]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    let mut data = tract_ndarray::ArrayD::<f32>::zeros(vec![2, 2, 2]);
    *data.as_slice_mut().unwrap().last_mut().unwrap() = 1.0;
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [2, 2])?,
        kernel_format: KernelFormat::HWIO,
        group: 2,
        data,
        kernel,
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_upper() -> anyhow::Result<()> {
    let data = ArrayD::zeros(vec![2, 1]);
    let kernel = ArrayD::zeros(vec![1, 1, 1]);
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2]).unwrap(),
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data,
        kernel,
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn explicit_dnn() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Explicit(tvec!(2), tvec!(0), false),
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn dnn_2d() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(1, 2, [1, 1]).unwrap(),
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr4(&[[[[0.0]], [[1.0]]]]).into_dyn(),
        kernel: arr4(&[[[[0.0]], [[1.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}
