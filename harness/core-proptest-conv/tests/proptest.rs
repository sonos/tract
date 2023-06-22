use core_proptest_conv::*;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::cnn::*;
use tract_core::ops::nn::*;
use tract_itertools::izip;
use tract_ndarray::*;

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
        let (shape_out, left_pads): (TVec<_>, TVec<_>) = if self.pad == PaddingSpec::Valid {
            izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides)
                .map(|(i, k, s)| {
                    let out = (*i + 1).saturating_sub(*k).divceil(*s);
                    (out, 0)
                })
                .unzip()
        } else {
            izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides)
                .map(|(input, k, stride)| {
                    let out = input.divceil(*stride);
                    let pad = ((out - 1) * stride + k).saturating_sub(*input);
                    (out, pad / 2)
                })
                .unzip()
        };
        let shape_out = self
            .shape_in
            .fmt
            .from_n_c_hw(self.shape_in.n().cloned().unwrap_or(1), co_per_g * self.group, shape_out)
            .unwrap();
        dbg!(&self.shape_in);
        dbg!(self.kernel.shape());
        dbg!(&shape_out);
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
                                eprintln!("g:{g} ci:{ci} i:{i} co:{co} k:{k}");
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

    fn tract(&self) -> TractResult<ArrayD<f32>> {
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
        let mut output =
            model.into_optimized()?.into_runnable()?.run(tvec![self.data.clone().into_tvalue()])?;
        output.remove(0).into_tensor().into_array::<f32>()
    }
}

impl Arbitrary for ConvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<ConvProblem>;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            data_format(),
            kernel_format(),
            prop_oneof![Just(PaddingSpec::Valid), Just(PaddingSpec::SameUpper)],
            1usize..=3,
            1usize..=4,
            1usize..=4,
            1usize..=3,
            (1usize..=3).prop_flat_map(shapes),
        )
            .prop_flat_map(|(df, kf, pad, n, mut ci0, co0, group, (mut ker_shape, data_shape))| {
                // FIXME in HWIO order, only regular and depthwise are supported
                if kf == KernelFormat::HWIO && group > 1 {
                    ci0 = 1;
                }
                let shape_in = df.from_n_c_hw(n, ci0 * group, data_shape).unwrap();
                let data_in = tensor(shape_in.shape.iter().cloned().collect());
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
                (Just((kf, pad, shape_in, group)), data_in, kernel, bias, strides)
            })
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

proptest::proptest! {
    #[test]
    fn prop(pb in any::<ConvProblem>()) {
        pb.tract().unwrap().into_tensor().close_enough(&pb.reference().into_tensor(), true).unwrap();
    }
}

#[test]
fn trivial_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0.0f32]]]).into_dyn(),
        kernel: arr4(&[[[[0.0f32]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn trivial_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 1, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[1.0f32]]]).into_dyn(),
        kernel: arr3(&[[[1.0f32]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn trivial_2() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[1.0f32], [0.0]]]).into_dyn(),
        kernel: arr3(&[[[1.0f32]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn trivial_3() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
        kernel: arr3(&[[[0.0f32], [1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn nchw_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0f32, 1.0]]]).into_dyn(),
        kernel: arr3(&[[[1f32]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn group_3() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_4() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: arr3(&[[[0.0f32]], [[0.0]], [[0.0]], [[1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_5() -> TractResult<()> {
    let pb = ConvProblem {
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
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_6() -> TractResult<()> {
    let pb = ConvProblem {
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
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_7() -> TractResult<()> {
    let pb = ConvProblem {
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
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_8() -> TractResult<()> {
    let pb = ConvProblem {
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
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_9() -> TractResult<()> {
    let pb = ConvProblem {
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
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_10() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [2, 1, 4])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ArrayD::<f32>::zeros(vec![2, 2, 1, 4]),
        kernel: ArrayD::from_elem(vec![4, 1, 1, 1, 2], 1.0f32),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_11() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: arr2(&[[0.0, 1.0]]).into_dyn(),
        kernel: arr3(&[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.0]]])
            .into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_12() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::HWIO,
        group: 2,
        data: arr2(&[[0.0, 0.0]]).into_dyn(),
        kernel: arr3(&[[[0.0], [0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_13() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::HWIO,
        group: 2,
        data: arr2(&[[0.0, 1.0]]).into_dyn(),
        kernel: arr3(&[[[0.0, 0.0], [1.0, 0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_bias_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ArrayD::<f32>::zeros(vec![1, 1, 2]),
        kernel: ArrayD::<f32>::zeros(vec![4, 1, 1]),
        bias: Some(ArrayD::<f32>::zeros(vec![4])),
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_bias_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: arr2(&[[0.0, 0.0]]).into_dyn(),
        kernel: ArrayD::<f32>::zeros(vec![4, 1, 1]),
        bias: Some(arr1(&[0.0, 0.0, 0.0, 1.0]).into_dyn()),
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn bias_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ArrayD::<f32>::zeros(vec![2, 1]),
        kernel: ArrayD::<f32>::zeros(vec![1, 1, 2]),
        bias: Some(ArrayD::<f32>::zeros(vec![1])),
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn bias_1() -> TractResult<()> {
    let kernel = ArrayD::<f32>::zeros(vec![2, 1, 2]);
    let data = ArrayD::<f32>::zeros(vec![2, 1]);
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
fn bias_chw_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr2(&[[0f32, 0., 0.]]).into_dyn(),
        kernel: arr3(&[[[0f32]], [[0.]], [[0.]]]).into_dyn(),
        bias: Some(arr1(&[0f32, 0., 1.]).into_dyn()),
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn batch_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(2, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ArrayD::<f32>::zeros(vec![2, 2, 1]),
        kernel: ArrayD::<f32>::zeros(vec![1, 1, 2]),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn batch_1() -> TractResult<()> {
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
fn batch_2() -> TractResult<()> {
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
fn bias_3d_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [1, 1, 2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ArrayD::<f32>::zeros(vec![1, 1, 1, 2]),
        kernel: ArrayD::<f32>::zeros(vec![1, 1, 1, 1, 1]),
        bias: Some(ArrayD::<f32>::ones(vec![1])),
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn batch_3d() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(1, 1, [2, 2, 1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ArrayD::<f32>::zeros(vec![1, 1, 2, 2, 1]),
        kernel: ArrayD::<f32>::zeros(vec![1, 1, 1, 1, 1]),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1, 1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ArrayD::<f32>::zeros(vec![1, 1]),
        kernel: ArrayD::<f32>::zeros(vec![1, 1, 1]),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr2(&[[0.0], [1.0]]).into_dyn(),
        kernel: arr3(&[[[0.0, 1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_2() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0.0], [0.0]], [[0.0], [1.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0], [0.0]]], [[[0.0], [1.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_2d_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0.0], [0.0], [1.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn same_2d_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0.0], [0.0]], [[1.0], [0.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0, 1.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr2(&[[0.0], [0.0]]).into_dyn(),
        kernel: arr3(&[[[0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr2(&[[0.0], [0.0], [1.0]]).into_dyn(),
        kernel: arr3(&[[[1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr2(&[[0.0], [0.0], [1.0]]).into_dyn(),
        kernel: arr3(&[[[0.0, 0.0, 1.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(3),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2d_same_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0.0], [0.0], [1.0]]]).into_dyn(),
        kernel: arr4(&[[[[1.0, 0.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2d_same_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ArrayD::<f32>::zeros(vec![2, 3, 1]),
        kernel: ArrayD::<f32>::zeros(vec![1, 1, 1, 3]),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_2d_same_2() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [2, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0.0], [0.0], [1.0]], [[0.0], [0.0], [0.0]]]).into_dyn(),
        kernel: arr4(&[[[[1.0, 0.0, 0.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::SameUpper,
        strides: tvec!(1, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn strides_two_axes() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1, 1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr3(&[[[0.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0]]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(2, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn lazy_im2col_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 1, [2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: arr2(&[[0.0, 0.0]]).into_dyn(),
        kernel: arr3(&[[[0.0, 0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn lazy_im2col_big() -> TractResult<()> {
    let mut kernel = ArrayD::<f32>::zeros(vec![1, 4, 1, 3, 2]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 4, [2, 5, 4])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ArrayD::<f32>::zeros(vec![4, 2, 5, 4]),
        kernel,
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(2, 2, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn lazy_im2col_big_2() -> TractResult<()> {
    let mut kernel = ArrayD::<f32>::zeros(vec![1, 4, 1, 3, 2]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 4, [2, 5, 4])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ArrayD::<f32>::zeros(vec![1, 2, 5, 4, 4]),
        kernel,
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(2, 3, 2),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn depthwise_0() -> TractResult<()> {
    let mut kernel = ArrayD::<f32>::zeros(vec![2, 2, 2, 1]);
    let len = kernel.len();
    kernel.as_slice_mut().unwrap()[len - 1] = 1.0;
    let mut data = ArrayD::<f32>::zeros(vec![2, 2, 2]);
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
fn same_upper() -> TractResult<()> {
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
fn dnn_2d() -> TractResult<()> {
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

#[test]
fn group_owhi_0() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [1]).unwrap(),
        kernel_format: KernelFormat::OHWI,
        group: 2,
        data: arr2(&[[0.0], [0.0]]).into_dyn(),
        kernel: arr3(&[[[0.0, 0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

/*
#[test]
fn group_owhi_1() -> TractResult<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 2, [1]).unwrap(),
        kernel_format: KernelFormat::OHWI,
        group: 2,
        data: arr2(&[[0.0], [1.0]]).into_dyn(),
        kernel: arr3(&[[[0.0, 0.0]], [[1.0, 0.0]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}
*/
