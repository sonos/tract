use super::proptest::*;
use crate::internal::*;
use crate::ops::cnn::KernelFormat::*;
use crate::ops::cnn::*;
use crate::ops::matmul::*;
use crate::ops::nn::DataFormat::*;
use crate::ops::nn::*;
use crate::setup_test_logger;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use tract_ndarray::prelude::*;
use tract_ndarray::*;

pub fn qtensor(shape: Vec<usize>) -> BoxedStrategy<ArrayD<i8>> {
    let len = shape.iter().product::<usize>();
    vec(any::<i8>(), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap())
        .boxed()
}

pub fn q_params() -> BoxedStrategy<QParams> {
    (-10i32..10, -10i32..10, -10i32..10, 0.001..10f32)
        .prop_map(|(a0, b0, c0, scale)| QParams {
            a0: AttrOrInput::Attr(rctensor0(a0)),
            b0: AttrOrInput::Attr(rctensor0(b0)),
            c0: AttrOrInput::Attr(rctensor0(c0)),
            a_scale: AttrOrInput::Attr(rctensor0(1f32)),
            b_scale: AttrOrInput::Attr(rctensor0(1f32)),
            c_scale: AttrOrInput::Attr(rctensor0(scale)),
        })
        .boxed()
}

#[derive(Debug)]
struct QConvProblem {
    shape_in: DataShape,
    shape_out: DataShape,
    kernel_format: KernelFormat,
    group: usize,
    data: ArrayD<i8>,
    kernel: ArrayD<i8>,
    bias: Option<ArrayD<i32>>,
    qp: QParams,
}

impl QConvProblem {
    fn geo_ker(&self) -> &[usize] {
        &self.kernel.shape()[self.kernel_format.h_axis()..][..self.shape_in.hw_rank()]
    }

    fn reference(&self) -> ArrayD<i8> {
        setup_test_logger();
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        assert_eq!(self.shape_out.fmt, self.shape_in.fmt);
        let n = *self.shape_in.n().clone().unwrap_or(&1);
        let ci_per_g = self.shape_in.c() / self.group;
        let co_per_g = self.shape_out.c() / self.group;
        let a0 = self.qp.a0.as_static().unwrap().cast_to_scalar::<i32>().unwrap();
        let b0 = self.qp.b0.as_static().unwrap().cast_to_scalar::<i32>().unwrap();
        let c0 = self.qp.c0.as_static().unwrap().cast_to_scalar::<i32>().unwrap();
        let scale = self.qp.c_scale.as_static().unwrap().cast_to_scalar::<f32>().unwrap();
        let mut temp = ArrayD::<i32>::zeros(&*self.shape_out.shape);
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
                            let i = self.data[&*input_coords] as i32;
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
            shape[self.shape_out.c_axis()] = bias.len();
            temp += &bias.clone().into_shape(shape).unwrap();
        }
        temp.mapv(|i| {
            (round_away(i as f32 / scale as f32) as i32 + c0)
                .max(std::i8::MIN as i32)
                .min(std::i8::MAX as i32) as i8
        })
    }

    fn tract(&self, optim: bool) -> anyhow::Result<ArrayD<i8>> {
        setup_test_logger();
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        let mut model = TypedModel::default();
        let wire = model
            .add_source("input", TypedFact::dt_shape(i8::datum_type(), &self.shape_in.shape))?;
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
            Some((i8::datum_type(), self.qp.clone())),
        );
        let wire = model.wire_node("conv", op, &[wire])?[0];
        model.set_output_outlets(&[wire])?;
        if optim {
            model = model.into_optimized()?;
        } else {
            model = model.declutter()?;
        }
        let mut output = model.into_runnable()?.run(tvec![self.data.clone().into_tensor()])?;
        Ok(output.remove(0).into_tensor().into_array::<i8>()?)
    }

    fn check(&self) -> TestCaseResult {
        prop_assert_eq!(self.tract(false).unwrap(), self.reference());
        prop_assert_eq!(self.tract(true).unwrap(), self.reference());
        Ok(())
    }
}

impl Arbitrary for QConvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<QConvProblem>;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            any::<DataFormat>(),
            any::<KernelFormat>(),
            1usize..=10,
            1usize..=8,
            1usize..=8,
            1usize..=1, // group, FIXME
            (1usize..=3).prop_flat_map(|r| shapes(r)),
            q_params(),
        )
            .prop_flat_map(|(df, kf, n, mut ci0, co0, group, (mut ker_shape, data_shape), qp)| {
                // FIXME in HWIO order, only regular and depthwise are supported
                if kf == KernelFormat::HWIO && group > 1 {
                    ci0 = 1;
                }
                let shape_in = df.from_n_c_hw(n, ci0 * group, &data_shape).unwrap();
                let shape_out: TVec<_> =
                    izip!(&ker_shape, data_shape).map(|(k, d)| d - k + 1).collect();
                let shape_out = df.from_n_c_hw(n, co0 * group, &shape_out).unwrap();
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
                };
                let kernel = qtensor(ker_shape);
                let bias = proptest::option::of(
                    qtensor(vec![co0 * group]).prop_map(|a| a.mapv(|v| v as i32)),
                );
                (Just((kf, shape_in, shape_out, group, qp)), data_in, kernel, bias)
                // FIXME
            })
            .prop_map(|((kernel_format, shape_in, shape_out, group, qp), data, kernel, bias)| {
                QConvProblem { shape_in, shape_out, kernel_format, group, data, kernel, bias, qp }
            })
            .boxed()
    }
}

proptest::proptest! {
    #[test]
    fn prop(pb in any::<QConvProblem>()) {
        pb.check().unwrap()
    }
}

#[test]
fn trivial_0() {
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr2(&[[0i8]]).into_dyn(),
        kernel: arr3(&[[[0i8]]]).into_dyn(),
        bias: None,
        qp: QParams::noop_static(i8::datum_type()),
    }
    .check()
    .unwrap();
}

#[test]
fn trivial_1() {
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr2(&[[2i8]]).into_dyn(),
        kernel: arr3(&[[[64i8]]]).into_dyn(),
        bias: None,
        qp: QParams::noop_static(i8::datum_type()),
    }
    .check()
    .unwrap();
}

#[test]
fn trivial_2() {
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 1, &[2]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr2(&[[-13i8], [26]]).into_dyn(),
        kernel: arr3(&[[[8i8, -2]]]).into_dyn(),
        bias: None,
        qp: QParams::noop_static(i8::datum_type()),
    }
    .check()
    .unwrap();
}

#[test]
fn shape_0() {
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 1, &[1, 2]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 1, &[1, 2]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr3(&[[[0], [0]]]).into_dyn(),
        kernel: arr4(&[[[[0]]]]).into_dyn(),
        bias: None,
        qp: QParams::noop_static(i8::datum_type()),
    }
    .check()
    .unwrap();
}

#[test]
fn batch_0() {
    QConvProblem {
        shape_in: NHWC.from_n_c_hw(3, 1, &[2]).unwrap(),
        shape_out: NHWC.from_n_c_hw(3, 1, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr3(&[[[0], [0]], [[0], [0]], [[0], [0]]]).into_dyn(),
        kernel: arr3(&[[[0, 0]]]).into_dyn(),
        bias: None,
        qp: QParams::noop_static(i8::datum_type()),
    }
    .check()
    .unwrap();
}

#[test]
fn a0_0() {
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr2(&[[1]]).into_dyn(),
        kernel: arr3(&[[[0]]]).into_dyn(),
        bias: None,
        qp: QParams::noop_static(i8::datum_type()),
    }
    .check()
    .unwrap();
}

#[test]
fn scale_0() {
    let mut qp = QParams::noop_static(i8::datum_type());
    qp.c_scale = AttrOrInput::Attr(rctensor0(9.274534f32));
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr2(&[[-1]]).into_dyn(),
        kernel: arr3(&[[[1]]]).into_dyn(),
        bias: None,
        qp,
    }
    .check()
    .unwrap();
}

#[test]
fn scale_1() {
    let mut qp = QParams::noop_static(i8::datum_type());
    qp.c_scale = AttrOrInput::Attr(rctensor0(1.1400417f32));
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr2(&[[41]]).into_dyn(),
        kernel: arr3(&[[[1]]]).into_dyn(),
        bias: None,
        qp,
    }
    .check().unwrap();
}

#[test]
#[ignore]
fn group_0() {
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 2, &[1]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 2, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 2,
        data: arr2(&[[0, 0]]).into_dyn(),
        kernel: arr3(&[[[0]], [[0]]]).into_dyn(),
        bias: None,
        qp: QParams::noop_static(i8::datum_type()),
    }
    .check()
    .unwrap();
}

#[test]
fn rounding_on_arm() {
    let mut qp = QParams::noop_static(i8::datum_type());
    qp.c_scale = AttrOrInput::Attr(rctensor0(1.3759452f32));
    QConvProblem {
        shape_in: HWC.from_n_c_hw(1, 1, &[1]).unwrap(),
        shape_out: HWC.from_n_c_hw(1, 2, &[1]).unwrap(),
        kernel_format: OIHW,
        group: 1,
        data: arr2(&[[1i8]]).into_dyn(),
        kernel: arr3(&[[[0i8]], [[-15]]]).into_dyn(),
        bias: None,
        qp,
    }
    .check()
    .unwrap();
}

fn test_conv_q_and_bias(a0: i32, b0: i32, c0: i32, c_scale: f32, k: i8, i: i8, bias: i32) {
    use super::*;
    let mut model = TypedModel::default();
    let source = model.add_source("input", TypedFact::dt_shape(i8::datum_type(), &[1, 1])).unwrap();
    let mut q_params = QParams::noop_static(i32::datum_type());
    q_params.a0 = AttrOrInput::Attr(rctensor0(a0));
    q_params.b0 = AttrOrInput::Attr(rctensor0(b0));
    q_params.c_scale = AttrOrInput::Attr(rctensor0(c_scale));
    q_params.c0 = AttrOrInput::Attr(rctensor0(c0));
    let conv = ConvUnary {
        pool_spec: PoolSpec {
            data_format: CHW,
            kernel_shape: tvec![1],
            padding: PaddingSpec::Valid,
            dilations: None,
            strides: None,
            output_channel_override: Some(1),
        },
        kernel_fmt: KernelFormat::OIHW,
        kernel: rctensor3(&[[[k]]]),
        group: 1,
        bias: Some(rctensor1(&[bias])),
        q_params: Some((i32::datum_type(), q_params)),
    };
    let output = model.wire_node("conv", conv, &[source]).unwrap();
    model.set_output_outlets(&output).unwrap();

    let input = tvec!(tensor2(&[[i]]));
    let expected =
        round_away((((k as i32) - a0) * ((i as i32) - b0) + bias) as f32 / c_scale) as i32 + c0;

    let expected = tensor2(&[[expected]]);

    //        dbg!(&model);
    let output = model.clone().into_runnable().unwrap().run(input.clone()).unwrap();
    assert_eq!(&*output[0], &expected);

    let output = model.declutter().unwrap().into_runnable().unwrap().run(input.clone()).unwrap();
    assert_eq!(&*output[0], &expected);

    let output = model.into_optimized().unwrap().into_runnable().unwrap().run(input).unwrap();
    assert_eq!(&*output[0], &expected);
}

fn round_away<F: num_traits::Float>(x: F) -> F {
    x.abs().round() * x.signum()
}

proptest::proptest! {
    #[test]
    fn conv_q_and_bias_prop(a0 in 0i32..5, b0 in 0i32..5, c0 in 0i32..5, c_scale in 0f32..1., k in 0i8..5, i in 0i8..5, bias in 0i32..5) {
        test_conv_q_and_bias(a0, b0, c0, c_scale, i, k, bias)
    }
}

#[test]
fn conv_q_and_bias_0() {
    test_conv_q_and_bias(0, 0, 0, 0.4447719, 0, 0, 1)
}

#[test]
fn conv_q_and_bias_1() {
    test_conv_q_and_bias(1, 0, 0, 0.4447719, 0, 1, 0)
}

#[test]
fn conv_q_and_bias_2() {
    test_conv_q_and_bias(4, 1, 0, 0.00029599667, 3, 0, 2)
}
