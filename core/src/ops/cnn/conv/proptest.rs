use super::KernelFormat;
use crate::internal::*;
use crate::ops::cnn::*;
use crate::ops::nn::*;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_ndarray::prelude::*;

#[derive(Debug)]
struct ConvProblem {
    shape_in: DataShape,
    shape_out: DataShape,
    kernel_format: KernelFormat,
    group: usize,
    data: ArrayD<f32>,
    kernel: ArrayD<f32>,
    bias: Option<ArrayD<f32>>,
}

impl ConvProblem {
    fn geo_ker(&self) -> &[usize] {
        &self.kernel.shape()[self.kernel_format.h_axis()..][..self.shape_in.hw_rank()]
    }

    fn reference(&self) -> ArrayD<f32> {
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        let mut out = ArrayD::zeros(&*self.shape_out.shape);
        let n = *self.shape_in.n().clone().unwrap_or(&1);
        let ci_per_g = self.shape_in.c() / self.group;
        let co_per_g = self.shape_out.c() / self.group;
        for n in 0..n {
            for g in 0..self.group {
                for geo_out in tract_ndarray::indices(self.shape_out.hw_dims()) {
                    let mut output_coords: TVec<usize> = geo_out.slice().into();
                    if self.shape_in.fmt.has_n() {
                        output_coords.insert(0, n);
                    }
                    output_coords.insert(self.shape_out.c_axis(), 0);
                    for geo_ker in tract_ndarray::indices(self.geo_ker()) {
                        let mut input_coords: TVec<usize> =
                            izip!(geo_out.slice(), geo_ker.slice()).map(|(a, b)| a + b).collect();
                        if self.shape_in.fmt.has_n() {
                            input_coords.insert(0, n);
                        }
                        input_coords.insert(self.shape_in.c_axis(), 0);
                        for ci in 0..ci_per_g {
                            input_coords[self.shape_in.c_axis()] = ci + g * ci_per_g;
                            let i = self.data[&*input_coords];
                            for co in 0..co_per_g {
                                output_coords[self.shape_out.c_axis()] = co + g * co_per_g;
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
            shape[self.shape_out.c_axis()] = bias.len();
            out += &bias.clone().into_shape(shape).unwrap();
        }
        out
    }

    fn tract(&self) -> anyhow::Result<ArrayD<f32>> {
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        let mut model = TypedModel::default();
        let wire = model
            .add_source("input", TypedFact::dt_shape(f32::datum_type(), &self.shape_in.shape))?;
        let op = ConvUnary::new(
            PoolSpec::new(
                self.shape_in.fmt,
                self.geo_ker().into(),
                PaddingSpec::Valid,
                None,
                None,
                Some(*self.shape_out.c()),
            ),
            self.kernel_format.clone(),
            self.kernel.clone().into_arc_tensor(),
            self.group,
            self.bias.clone().map(|a| a.into_arc_tensor()),
            None,
        );
        let wire = model.wire_node("conv", op, &[wire])?[0];
        model.set_output_outlets(&[wire])?;
        let mut output = dbg!(model.into_optimized()?)
            .into_runnable()?
            .run(tvec![self.data.clone().into_tensor()])?;
        Ok(output.remove(0).into_tensor().into_array::<f32>()?)
    }
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
            (1usize..=3).prop_flat_map(|r| shapes(r)),
        )
            .prop_flat_map(|(df, kf, n, mut ci0, co0, group, (mut ker_shape, data_shape))| {
                // FIXME in HWIO order, only regular and depthwise are supported
                if kf == KernelFormat::HWIO && group > 1 {
                    ci0 = 1;
                }
                let shape_in = df.from_n_c_hw(n, ci0 * group, &data_shape).unwrap();
                let shape_out: TVec<_> =
                    izip!(&ker_shape, data_shape).map(|(k, d)| d - k + 1).collect();
                let shape_out = df.from_n_c_hw(n, co0 * group, &shape_out).unwrap();
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
                };
                let kernel = tensor(ker_shape);
                let bias = proptest::option::of(tensor(vec![co0 * group]));
                (Just((kf, shape_in, shape_out, group)), data_in, kernel, bias)
            })
            .prop_map(|((kernel_format, shape_in, shape_out, group), data, kernel, bias)| {
                ConvProblem { shape_in, shape_out, kernel_format, group, data, kernel, bias }
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

fn tensor(shape: Vec<usize>) -> BoxedStrategy<ArrayD<f32>> {
    let len = shape.iter().product::<usize>();
    vec(any::<i8>().prop_map(|i| i as f32), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap())
        .boxed()
}

fn shapes(rank: usize) -> BoxedStrategy<(Vec<usize>, Vec<usize>)> {
    vec((1usize..3, 0usize..3).prop_map(|(k, exceed)| (k, k + exceed)), rank..=rank)
        .prop_map(|v| v.into_iter().unzip())
        .boxed()
}

proptest::proptest! {
    #[test]
    fn prop(pb in any::<ConvProblem>()) {
        prop_assert_eq!(pb.tract().unwrap(), pb.reference());
    }
}

#[test]
fn trivial_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 1, &[1])?,
        shape_out: DataFormat::NHWC.from_n_c_hw(1, 1, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn trivial_2() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 1, &[2])?,
        shape_out: DataFormat::NHWC.from_n_c_hw(1, 1, &[2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[1.0f32], [0.0]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn trivial_3() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, &[1])?,
        shape_out: DataFormat::NHWC.from_n_c_hw(1, 1, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32], [1.0]]]).into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}

#[test]
fn nchw_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(1, 1, &[2])?,
        shape_out: DataFormat::NCHW.from_n_c_hw(1, 1, &[2])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::arr3(&[[[0f32, 1.0]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1f32]]]).into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_1() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, &[1])?,
        shape_out: DataFormat::HWC.from_n_c_hw(1, 2, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
        bias: None,
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
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, &[1])?,
        shape_out: DataFormat::HWC.from_n_c_hw(1, 2, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32]], [[1.0]]]).into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_4() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, &[1])?,
        shape_out: DataFormat::HWC.from_n_c_hw(1, 4, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 1.0]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32]], [[0.0]], [[0.0]], [[1.0]]]).into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_5() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, &[1, 1])?,
        shape_out: DataFormat::HWC.from_n_c_hw(1, 4, &[1, 1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
        kernel: tensor4(&[[[[0.0f32]]], [[[0.0]]], [[[0.0]]], [[[0.0]]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_6() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, &[1])?,
        shape_out: DataFormat::NHWC.from_n_c_hw(1, 4, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
        kernel: tensor3(&[[[0.0f32]], [[0.0]], [[0.0]], [[0.0]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_7() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NCHW.from_n_c_hw(1, 2, &[2])?,
        shape_out: DataFormat::NCHW.from_n_c_hw(1, 4, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr3(&[[[0.0f32, 0.0], [0.0, 1.0]]]).into_dyn(),
        kernel: tensor3(&[[[0.0f32, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 1.0]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_8() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 4, &[1])?,
        shape_out: DataFormat::HWC.from_n_c_hw(1, 2, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 0.0, 0.0, 1.0]]).into_dyn(),
        kernel: tensor3(&[[[0.0f32], [0.0]], [[0.0], [0.0]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_9() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 2, &[2])?,
        shape_out: DataFormat::HWC.from_n_c_hw(1, 4, &[2])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::arr2(&[[0.0f32, 0.0], [0.0, 1.0]]).into_dyn(),
        kernel: tensor3(&[[[0.0f32]], [[0.0]], [[0.0]], [[1.0]]])
            .into_array::<f32>()
            .unwrap()
            .into_dyn(),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_10() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::CHW.from_n_c_hw(1, 2, &[2, 1, 4])?,
        shape_out: DataFormat::CHW.from_n_c_hw(1, 4, &[2, 1, 3])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::ArrayD::<f32>::zeros(vec![2, 2, 1, 4]),
        kernel: ndarray::ArrayD::from_elem(vec![4, 1, 1, 1, 2], 1.0f32),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn group_bias_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(1, 2, &[1])?,
        shape_out: DataFormat::NHWC.from_n_c_hw(1, 4, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 2,
        data: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![4, 1, 1]),
        bias: Some(ndarray::ArrayD::<f32>::zeros(vec![4])),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn bias_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, &[2])?,
        shape_out: DataFormat::HWC.from_n_c_hw(1, 1, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::ArrayD::<f32>::zeros(vec![2, 1]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2]),
        bias: Some(ndarray::ArrayD::<f32>::zeros(vec![1])),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

#[test]
fn batch_0() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::NHWC.from_n_c_hw(2, 1, &[2])?,
        shape_out: DataFormat::NHWC.from_n_c_hw(2, 1, &[1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: ndarray::ArrayD::<f32>::zeros(vec![2, 2, 1]),
        kernel: ndarray::ArrayD::<f32>::zeros(vec![1, 1, 2]),
        bias: None,
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}
