use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
// use proptest::*;
use tract_core::internal::*;
use tract_core::ops::cnn::KernelFormat::*;
use tract_core::ops::cnn::{ConvUnary, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::math::round_ties_to_even;
use tract_core::ops::nn::DataFormat::*;
use tract_core::ops::nn::DataShape;
use tract_itertools::izip;
use tract_ndarray::*;

pub fn qtensor(shape: Vec<usize>) -> BoxedStrategy<ArrayD<i8>> {
    let len = shape.iter().product::<usize>();
    vec(any::<i8>(), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap())
        .boxed()
}

pub fn q_params() -> BoxedStrategy<[Tensor; 6]> {
    (-10i32..10, -10i32..10, -10i32..10, -3..3i32, -3..3i32, -3..3i32)
        .prop_map(|(a0, b0, c0, a_scale, b_scale, c_scale)| {
            [
                tensor0(a0),
                tensor0(2f32.powi(a_scale)),
                tensor0(b0),
                tensor0(2f32.powi(b_scale)),
                tensor0(c0),
                tensor0(2f32.powi(c_scale)),
            ]
        })
        .boxed()
}

#[derive(Debug, Clone)]
struct QConvProblem {
    shape_in: DataShape,
    kernel_format: KernelFormat,
    co: usize,
    group: usize,
    data: ArrayD<i8>,
    kernel: ArrayD<i8>,
    bias: Option<ArrayD<i32>>,
    qp: [Tensor; 6],
}

impl QConvProblem {
    fn geo_ker(&self) -> &[usize] {
        &self.kernel.shape()[self.kernel_format.h_axis()..][..self.shape_in.hw_rank()]
    }

    fn reference(&self) -> ArrayD<i8> {
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        let n = *self.shape_in.n().unwrap_or(&1);
        let ci_per_g = self.shape_in.c() / self.group;
        let co_per_g = self.co / self.group;
        let a0 = self.qp[0].cast_to_scalar::<i32>().unwrap();
        let b0 = self.qp[2].cast_to_scalar::<i32>().unwrap();
        let c0 = self.qp[4].cast_to_scalar::<i32>().unwrap();
        let scale = self.qp[5].cast_to_scalar::<f32>().unwrap()
            / self.qp[1].cast_to_scalar::<f32>().unwrap()
            / self.qp[3].cast_to_scalar::<f32>().unwrap();
        let shape_out: TVec<usize> = izip!(self.shape_in.hw_dims(), self.geo_ker())
            .map(|(i, k)| (*i + 1).saturating_sub(*k))
            .collect();
        let shape_out = self
            .shape_in
            .fmt
            .from_n_c_hw(self.shape_in.n().cloned().unwrap_or(1), co_per_g * self.group, shape_out)
            .unwrap();
        let mut temp = ArrayD::<i32>::zeros(&*shape_out.shape);
        for n in 0..n {
            for g in 0..self.group {
                for geo_out in tract_ndarray::indices(shape_out.hw_dims()) {
                    let mut output_coords: TVec<usize> = geo_out.slice().into();
                    if self.shape_in.fmt.has_n() {
                        output_coords.insert(0, n);
                    }
                    output_coords.insert(shape_out.c_axis(), 0);
                    for geo_ker in tract_ndarray::indices(self.geo_ker()) {
                        let mut input_coords: TVec<usize> =
                            izip!(geo_out.slice(), geo_ker.slice()).map(|(a, b)| a + b).collect();
                        if self.shape_in.fmt.has_n() {
                            input_coords.insert(0, n);
                        }
                        input_coords.insert(self.shape_in.c_axis(), 0);
                        for ci in 0..ci_per_g {
                            input_coords[self.shape_in.c_axis()] = ci + g * ci_per_g;
                            let i = self.data[&*input_coords] as i32;
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
                                let k = self.kernel[&*kernel_coords] as i32;
                                temp[&*output_coords] += (k - a0) * (i - b0);
                            }
                        }
                    }
                }
            }
        }
        if let Some(bias) = &self.bias {
            let mut shape = vec![1; temp.ndim()];
            shape[shape_out.c_axis()] = bias.len();
            temp += &bias.clone().into_shape(shape).unwrap();
        }
        temp.mapv(|i| {
            (round_ties_to_even(i as f32 / scale) as i32 + c0)
                .max(std::i8::MIN as i32)
                .min(std::i8::MAX as i32) as i8
        })
    }

    fn tract(&self) -> TractResult<TypedModel> {
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        let mut model = TypedModel::default();
        let wire = model.add_source("input", i8::fact(&self.shape_in.shape))?;
        let mut inputs = tvec!(wire);
        for (ix, qp) in self.qp.iter().enumerate() {
            inputs.push(model.add_const(format!("qp.{ix}"), qp.clone())?);
        }
        let op = ConvUnary::new(
            PoolSpec::new(
                self.shape_in.fmt,
                self.geo_ker().into(),
                PaddingSpec::Valid,
                None,
                None,
                Some(self.co),
            ),
            self.kernel_format,
            self.kernel.clone().into_arc_tensor(),
            self.group,
            self.bias.clone().map(|a| a.into_arc_tensor()),
            Some(i8::datum_type()),
        );
        let wire = model.wire_node("conv", op, &inputs)?[0];
        model.set_output_outlets(&[wire])?;
        Ok(model)
    }
}

impl Test for QConvProblem {
    fn run(&self, runtime: &dyn Runtime) -> TractResult<()> {
        let model = runtime.prepare(self.tract()?)?;
        let output = model.run(tvec!(self.data.clone().into_tensor().into_tvalue()))?.remove(0);
        output.close_enough(&self.reference().into_tensor(), Approximation::Exact)
    }
}

impl Arbitrary for QConvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<QConvProblem>;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            crate::data_format(),
            crate::kernel_format(),
            1usize..=10,
            1usize..=8,
            1usize..=8,
            1usize..=3,
            (1usize..=3).prop_flat_map(crate::shapes),
            q_params(),
        )
            .prop_flat_map(|(df, kf, n, mut ci0, co0, group, (mut ker_shape, data_shape), qp)| {
                // FIXME in HWIO order, only regular and depthwise are supported
                if kf == KernelFormat::HWIO && group > 1 {
                    ci0 = 1;
                }
                let shape_in = df.from_n_c_hw(n, ci0 * group, data_shape).unwrap();
                let data_in = qtensor(shape_in.shape.iter().cloned().collect());
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
                        ker_shape.push(ci0 * group)
                    }
                };
                let kernel = qtensor(ker_shape);
                let bias = proptest::option::of(
                    qtensor(vec![co0 * group]).prop_map(|a| a.mapv(|v| v as i32)),
                );
                (Just((kf, shape_in, co0 * group, group, qp)), data_in, kernel, bias)
                // FIXME
            })
            .prop_map(|((kernel_format, shape_in, co, group, qp), data, kernel, bias)| {
                QConvProblem { shape_in, co, kernel_format, group, data, kernel, bias, qp }
            })
            .boxed()
    }
}

/*
proptest::proptest! {
    #[test]
    fn prop(pb in any::<QConvProblem>()) {
        pb.check().unwrap()
    }
}
*/

fn qp_noop_i8() -> [Tensor; 6] {
    [tensor0(0i8), tensor0(1f32), tensor0(0i8), tensor0(1f32), tensor0(0i8), tensor0(1f32)]
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add(
        "trivial_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[0i8]]).into_dyn(),
            kernel: arr3(&[[[0i8]]]).into_dyn(),
            bias: None,
            qp: qp_noop_i8(),
        },
    );
    suite.add(
        "trivial_1",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[2i8]]).into_dyn(),
            kernel: arr3(&[[[64i8]]]).into_dyn(),
            bias: None,
            qp: qp_noop_i8(),
        },
    );
    suite.add(
        "trivial_2",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [2]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[-13i8], [26]]).into_dyn(),
            kernel: arr3(&[[[8i8, -2]]]).into_dyn(),
            bias: None,
            qp: qp_noop_i8(),
        },
    );
    suite.add(
        "trivial_3",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 2,
            kernel_format: HWIO,
            group: 1,
            data: arr2(&[[0i8], [0]]).into_dyn(),
            kernel: arr3(&[[[0i8, 0], [0, 0]]]).into_dyn(),
            bias: None,
            qp: qp_noop_i8(),
        },
    );
    let mut qp = qp_noop_i8();
    qp[2] = tensor0(1i32);
    suite.add(
        "b0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            kernel_format: OIHW,
            co: 1,
            group: 1,
            data: arr2(&[[0i8]]).into_dyn(),
            kernel: arr3(&[[[-1i8]]]).into_dyn(),
            bias: None,
            qp,
        },
    );
    suite.add(
        "shape_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1, 2]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr3(&[[[0], [0]]]).into_dyn(),
            kernel: arr4(&[[[[0]]]]).into_dyn(),
            bias: None,
            qp: qp_noop_i8(),
        },
    );
    suite.add(
        "batch_0",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(3, 1, [2]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr3(&[[[0], [0]], [[0], [0]], [[0], [0]]]).into_dyn(),
            kernel: arr3(&[[[0, 0]]]).into_dyn(),
            bias: None,
            qp: qp_noop_i8(),
        },
    );
    let qp = qp_noop_i8();
    let data = ArrayD::zeros(vec![2, 1, 1]);
    let kernel = arr3(&[[[1]]]).into_dyn();
    suite.add(
        "batch_1",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(2, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data,
            kernel,
            bias: None,
            qp,
        },
    );
    suite.add(
        "a0_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[1]]).into_dyn(),
            kernel: arr3(&[[[0]]]).into_dyn(),
            bias: None,
            qp: qp_noop_i8(),
        },
    );
    let mut qp = qp_noop_i8();
    qp[5] = tensor0(9.274534f32);
    suite.add(
        "scale_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[-1]]).into_dyn(),
            kernel: arr3(&[[[1]]]).into_dyn(),
            bias: None,
            qp,
        },
    );
    let mut qp = qp_noop_i8();
    qp[5] = tensor0(1.1400417f32);
    suite.add(
        "scale_1",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[41]]).into_dyn(),
            kernel: arr3(&[[[1]]]).into_dyn(),
            bias: None,
            qp,
        },
    );
    let mut qp = qp_noop_i8();
    qp[3] = tensor0(0.5f32);
    suite.add(
        "scale_2",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[-1]]).into_dyn(),
            kernel: arr3(&[[[2]]]).into_dyn(),
            bias: None,
            qp,
        },
    );
    let mut qp = qp_noop_i8();
    qp[3] = tensor0(0.5f32);
    qp[5] = tensor0(2f32);
    suite.add(
        "scale_3",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[0i8]]).into_dyn(),
            kernel: arr3(&[[[0i8]]]).into_dyn(),
            bias: Some(arr1(&[35i32]).into_dyn()),
            qp,
        },
    );
    let mut qp = qp_noop_i8();
    qp[4] = tensor0(1i32);
    suite.add(
        "c0_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[0i8]]).into_dyn(),
            kernel: arr3(&[[[0i8]]]).into_dyn(),
            bias: None,
            qp,
        },
    );
    suite.add(
        "group_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 2,
            data: arr2(&[[0, 0]]).into_dyn(),
            kernel: arr3(&[[[0]], [[0]]]).into_dyn(),
            bias: None,
            qp: qp_noop_i8(),
        },
    );
    let mut qp = qp_noop_i8();
    qp[2] = tensor0(1i32);
    suite.add(
        "group_1",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 2,
            data: arr3(&[[[0], [0]]]).into_dyn(),
            kernel: arr3(&[[[1]], [[0]]]).into_dyn(),
            bias: None,
            qp,
        },
    );
    let mut qp = qp_noop_i8();
    qp[2] = tensor0(1i32);
    suite.add(
        "group_2",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 2,
            data: arr2(&[[0, 0]]).into_dyn(),
            kernel: arr3(&[[[0]], [[1]]]).into_dyn(),
            bias: None,
            qp,
        },
    );
    let mut qp = qp_noop_i8();
    qp[5] = tensor0(1.3759452f32);
    suite.add(
        "rounding_on_arm",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: arr2(&[[1i8]]).into_dyn(),
            kernel: arr3(&[[[0i8]], [[-15]]]).into_dyn(),
            bias: None,
            qp,
        },
    );
    let qp = qp_noop_i8();
    let data = ArrayD::zeros(vec![1, 1, 1, 1]);
    let kernel = ArrayD::zeros(vec![2, 1, 1, 1]);
    suite.add(
        "bias_1",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(1, 1, [1, 1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data,
            kernel,
            bias: Some(tract_ndarray::arr1(&[1, 2]).into_dyn()),
            qp,
        },
    );
    let qp = qp_noop_i8();
    let data = ArrayD::zeros(vec![1, 1]);
    let kernel = ArrayD::zeros(vec![2, 1, 1]);
    suite.add(
        "bias_2",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data,
            kernel,
            bias: Some(tract_ndarray::arr1(&[0, 1]).into_dyn()),
            qp,
        },
    );
    let mut qp = qp_noop_i8();
    qp[2] = tensor0(-1);
    let data = ArrayD::zeros(vec![2, 1]);
    let mut kernel = ArrayD::zeros(vec![5, 1, 2]);
    *kernel.as_slice_mut().unwrap().last_mut().unwrap() = -1;
    suite.add(
        "bias_3",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [2]).unwrap(),
            co: 5,
            kernel_format: OIHW,
            group: 1,
            data,
            kernel,
            bias: Some(ArrayD::zeros([5].as_ref())),
            qp,
        },
    );
    let qp = qp_noop_i8();
    let data = ArrayD::zeros(vec![1, 1]);
    let kernel = ArrayD::zeros(vec![2, 1, 1]);
    suite.add(
        "bias_in_chw",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data,
            kernel,
            bias: Some(ArrayD::zeros([2].as_ref())),
            qp,
        },
    );
    let qp = qp_noop_i8();
    let data = ArrayD::zeros(vec![1, 1, 1]);
    let kernel = ArrayD::zeros(vec![1, 1, 1]);
    suite.add(
        "bias_with_batch",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data,
            kernel,
            bias: Some(arr1(&[1]).into_dyn()),
            qp,
        },
    );
    let qp = qp_noop_i8();
    let data = ArrayD::zeros(vec![1, 1, 1]);
    let kernel = ArrayD::zeros(vec![2, 1, 1]);
    suite.add(
        "bias_vec_with_batch",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data,
            kernel,
            bias: Some(arr1(&[0, 1]).into_dyn()),
            qp,
        },
    );
    let qp = qp_noop_i8();
    let data = ArrayD::zeros(vec![1, 2]);
    let kernel = ArrayD::zeros(vec![5, 2, 1]);
    suite.add(
        "asan_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 5,
            kernel_format: OIHW,
            group: 1,
            data,
            kernel,
            bias: None,
            qp,
        },
    );
    Ok(suite)
}
