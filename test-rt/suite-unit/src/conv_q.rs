use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::cnn::KernelFormat::*;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::math::round_ties_to_even;
use tract_core::ops::nn::DataFormat::*;
use tract_core::ops::nn::DataShape;
use tract_core::tract_data::itertools::Itertools;
use tract_itertools::izip;
use tract_ndarray::*;

use crate::conv_f32::ConvProblemParams;
use crate::q_helpers::qtensor;

/* https://www.tensorflow.org/lite/performance/quantization_spec
CONV_2D
Input 0:
data_type  : int8
range      : [-128, 127]
granularity: per-tensor
Input 1 (Weight):
data_type  : int8
range      : [-127, 127]
granularity: per-axis (dim = 0)
restriction: zero_point = 0
Input 2 (Bias):
data_type  : int32
range      : [int32_min, int32_max]
granularity: per-axis
restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
Output 0:
data_type  : int8
range      : [-128, 127]
granularity: per-tensor
*/
#[allow(clippy::arc_with_non_send_sync)]
pub fn q_params(
    params: &QConvProblemParams,
    co: usize,
    kdt: DatumType,
    idt: DatumType,
    odt: DatumType,
) -> BoxedStrategy<[Tensor; 6]> {
    let params = params.clone();
    let per_channel =
        if params.tflite_rules && (kdt.is_unsigned() || idt.is_unsigned() || odt.is_unsigned()) {
            Just(false).boxed()
        } else {
            any::<bool>().boxed()
        };
    per_channel
        .prop_flat_map(move |per_channel| {
            let k0 = if per_channel && params.tflite_rules {
                Just(0i32).boxed()
            } else if kdt.is_signed() {
                (-10..10i32).boxed()
            } else {
                (0..20i32).boxed()
            };
            let x0 = if idt.is_signed() { -10i32..10i32 } else { 0..20 };
            let y0 = if odt.is_signed() { -10i32..10i32 } else { 0..20 };
            let k_scale_len = if per_channel { co } else { 1 };
            let k_scale = vec(-3..3i32, k_scale_len..=k_scale_len);
            (Just(per_channel), k0, x0, y0, k_scale, -3..3i32, -3..3i32)
        })
        .prop_map(|(per_channel, k0, x0, y0, k_scale, x_scale, y_scale)| {
            let k_scale_values = k_scale.iter().map(|x| 2f32.powi(*x)).collect_vec();
            [
                tensor0(x0),
                tensor0(2f32.powi(x_scale)),
                tensor0(k0),
                if per_channel { tensor1(&k_scale_values) } else { tensor0(k_scale_values[0]) },
                tensor0(y0),
                tensor0(2f32.powi(y_scale)),
            ]
        })
        .boxed()
}

#[derive(Debug, Clone, Default)]
pub struct QConvProblemParams {
    pub conv: ConvProblemParams,
    pub tflite_rules: bool,
}

#[derive(Debug, Clone)]
pub struct QConvProblem {
    pub shape_in: DataShape,
    pub kernel_format: KernelFormat,
    pub co: usize,
    pub group: usize,
    pub kernel: Tensor,
    pub bias: Option<Array1<i32>>,
    pub data: Tensor,
    pub qp: [Tensor; 6],
    pub raw_output_dt: DatumType,
}

impl QConvProblem {
    fn geo_ker(&self) -> &[usize] {
        &self.kernel.shape()[self.kernel_format.h_axis()..][..self.shape_in.hw_rank()]
    }

    fn reference(&self) -> Tensor {
        assert!(self.data.datum_type().size_of() == 1);
        assert!(self.kernel.datum_type().size_of() == 1);
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        let n = *self.shape_in.n().unwrap_or(&1);
        let ci_per_g = self.shape_in.c() / self.group;
        let co_per_g = self.co / self.group;
        let x0 = *self.qp[0].to_scalar::<i32>().unwrap();
        let k0 = *self.qp[2].to_scalar::<i32>().unwrap();
        let y0 = *self.qp[4].to_scalar::<i32>().unwrap();
        let x_scale = self.qp[1].cast_to_scalar::<f32>().unwrap();
        let y_scale = self.qp[5].cast_to_scalar::<f32>().unwrap();
        let kdt = self.kernel.datum_type();
        let idt = self.data.datum_type();
        let odt = self.raw_output_dt;
        assert!(k0 <= kdt.unquantized().max_value().cast_to_scalar::<i32>().unwrap());
        assert!(k0 >= kdt.unquantized().min_value().cast_to_scalar::<i32>().unwrap());
        assert!(x0 <= idt.unquantized().max_value().cast_to_scalar::<i32>().unwrap());
        assert!(x0 >= idt.unquantized().min_value().cast_to_scalar::<i32>().unwrap());
        assert!(y0 <= odt.unquantized().max_value().cast_to_scalar::<i32>().unwrap());
        assert!(y0 >= odt.unquantized().min_value().cast_to_scalar::<i32>().unwrap());
        let shape_out: TVec<usize> = izip!(self.shape_in.hw_dims(), self.geo_ker())
            .map(|(i, k)| (*i + 1).saturating_sub(*k))
            .collect();
        let shape_out = self
            .shape_in
            .fmt
            .from_n_c_hw(self.shape_in.n().cloned().unwrap_or(1), co_per_g * self.group, shape_out)
            .unwrap();
        // a is the kernel, it can be quantized per O axis
        let k_scale = if self.qp[3].len() == 1 {
            vec![self.qp[3].cast_to_scalar::<f32>().unwrap(); *shape_out.c()]
        } else {
            self.qp[3].as_slice::<f32>().unwrap().into()
        };
        let mut temp = ArrayD::<i32>::zeros(&*shape_out.shape);
        let data = self.data.cast_to::<i32>().unwrap();
        let data = data.to_array_view::<i32>().unwrap();
        let kernel = self.kernel.cast_to::<i32>().unwrap();
        let kernel = kernel.to_array_view::<i32>().unwrap();
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
                            let i = data[&*input_coords];
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
                                let k = kernel[&*kernel_coords];
                                temp[&*output_coords] += (k - k0) * (i - x0);
                            }
                        }
                    }
                }
            }
        }
        if let Some(bias) = &self.bias {
            let mut shape = vec![1; temp.ndim()];
            shape[shape_out.c_axis()] = bias.len();
            temp += &bias.clone().into_shape_with_order(shape).unwrap();
        }
        let cdt = self.output_dt();
        temp.axis_iter_mut(Axis(shape_out.c_axis())).zip(k_scale).for_each(
            |(mut view, k_scale)| {
                view.mapv_inplace(|i| {
                    (round_ties_to_even(i as f32 / y_scale * k_scale * x_scale) as i32 + y0)
                        .max(cdt.unquantized().min_value().cast_to_scalar::<i32>().unwrap())
                        .min(cdt.unquantized().max_value().cast_to_scalar::<i32>().unwrap())
                });
            },
        );
        let mut tensor = temp.into_tensor().cast_to_dt(cdt.unquantized()).unwrap().into_owned();
        unsafe { tensor.set_datum_type(cdt) };
        tensor
    }

    fn output_dt(&self) -> DatumType {
        self.raw_output_dt.quantize(QParams::ZpScale {
            zero_point: self.qp[4].cast_to_scalar().unwrap(),
            scale: *self.qp[5].to_scalar().unwrap(),
        })
    }

    fn tract(&self) -> TractResult<TypedModel> {
        assert!(self.data.shape() == &*self.shape_in.shape);
        let mut model = TypedModel::default();
        let idt = self.data.datum_type().quantize(QParams::ZpScale {
            zero_point: self.qp[0].cast_to::<i32>()?.as_slice::<i32>()?[0],
            scale: self.qp[1].cast_to::<f32>()?.as_slice::<f32>()?[0],
        });
        let kdt = self.kernel.datum_type().quantize(QParams::ZpScale {
            zero_point: self.qp[2].cast_to::<i32>()?.as_slice::<i32>()?[0],
            scale: self.qp[3].cast_to::<f32>()?.as_slice::<f32>()?[0],
        });
        let wire = model.add_source("input", idt.fact(&self.shape_in.shape))?;
        let mut inputs = tvec!(wire);
        let mut kernel = self.kernel.clone().into_tensor();
        unsafe { kernel.set_datum_type(kdt) };
        inputs.push(model.add_const("kernel", kernel.into_arc_tensor())?);
        let bias = if let Some(bias) = &self.bias {
            bias.clone().into_arc_tensor()
        } else {
            rctensor0(0i32)
        };
        inputs.push(model.add_const("bias", bias)?);
        for (ix, qp) in self.qp.iter().enumerate() {
            inputs.push(model.add_const(format!("qp.{ix}"), qp.clone())?);
        }
        let op = Conv::new(
            PoolSpec::new(
                self.shape_in.fmt,
                self.geo_ker().into(),
                PaddingSpec::Valid,
                None,
                None,
                *self.shape_in.c(),
                self.co,
            ),
            self.kernel_format,
            self.group,
            Some(self.output_dt()),
        );
        let wire = model.wire_node("conv", op, &inputs)?[0];
        model.set_output_outlets(&[wire])?;
        Ok(model)
    }
}

impl Test for QConvProblem {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let reference = self.reference();
        let mut model = self.tract().context("Building model")?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let model = runtime.prepare(model).context("Preparing model")?;
        let idt = self.data.datum_type().quantize(QParams::ZpScale {
            zero_point: self.qp[0].cast_to_scalar()?,
            scale: *self.qp[1].to_scalar()?,
        });
        let data = self.data.clone().into_tensor().cast_to_dt(idt)?.into_owned().into_tvalue();
        let output = model.run(tvec!(data))?.remove(0);
        eprintln!("reference: {reference:?}\noutput   : {output:?}");
        output.close_enough(&reference, approx)
    }
}

impl Arbitrary for QConvProblem {
    type Parameters = QConvProblemParams;
    type Strategy = BoxedStrategy<QConvProblem>;
    fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
        let geo_rank = params.conv.geo_rank.clone().unwrap_or(1..4);
        (
            crate::data_format(),
            crate::kernel_format(),
            1usize..=10,
            1usize..=8,
            1usize..=8,
            1usize..=(if params.conv.no_group { 1 } else { 3 }),
            geo_rank.prop_flat_map(crate::shapes),
            prop_oneof![Just(DatumType::I8), Just(DatumType::U8)],
            prop_oneof![Just(DatumType::I8), Just(DatumType::U8)],
            prop_oneof![Just(DatumType::I8), Just(DatumType::U8), Just(DatumType::I32)],
        )
            .prop_flat_map(
                move |(
                    df,
                    kf,
                    n,
                    mut ci0,
                    mut co0,
                    group,
                    (mut ker_shape, data_shape),
                    kdt,
                    idt,
                    odt,
                )| {
                    // FIXME in HWIO order, only regular and depthwise are supported
                    if params.conv.no_arbitrary_grouping && group > 1 {
                        ci0 = 1;
                        co0 = 1;
                    }
                    if kf == KernelFormat::HWIO && group > 1 {
                        ci0 = 1;
                    }
                    let qp = q_params(&params, co0 * group, kdt, idt, odt);
                    let shape_in = df.from_n_c_hw(n, ci0 * group, data_shape).unwrap();
                    let data_in = qtensor(shape_in.shape.iter().cloned().collect(), idt);
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
                    let kernel = qtensor(ker_shape, kdt);
                    let bias = proptest::option::of(
                        qtensor(vec![co0 * group], i32::datum_type()).prop_map(|b| {
                            arr1(b.cast_to::<i32>().unwrap().as_slice::<i32>().unwrap())
                        }),
                    );
                    (Just((kf, shape_in, co0 * group, group, odt)), data_in, kernel, bias, qp)
                },
            )
            .prop_map(
                |((kernel_format, shape_in, co, group, raw_output_dt), data, kernel, bias, qp)| {
                    QConvProblem {
                        shape_in,
                        co,
                        kernel_format,
                        group,
                        data: data.into_tensor(),
                        kernel: kernel.into_tensor(),
                        bias,
                        qp,
                        raw_output_dt,
                    }
                },
            )
            .boxed()
    }
}

fn qp_noop_i8() -> [Tensor; 6] {
    [tensor0(0i32), tensor0(1f32), tensor0(0i32), tensor0(1f32), tensor0(0i32), tensor0(1f32)]
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<QConvProblem>("proptest", QConvProblemParams::default());

    suite.add(
        "trivial_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[0i8]]),
            kernel: tensor3(&[[[0i8]]]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    suite.add(
        "trivial_1",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[2i8]]),
            kernel: tensor3(&[[[64i8]]]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    suite.add(
        "trivial_2",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [2]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[-13i8], [26]]),
            kernel: tensor3(&[[[8i8, -2]]]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    suite.add(
        "trivial_3",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 2,
            kernel_format: HWIO,
            group: 1,
            data: tensor2(&[[0i8], [0]]),
            kernel: tensor3(&[[[0i8, 0], [0, 0]]]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    suite.add(
        "trivial_4",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 8,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[0i8], [1i8]]),
            kernel: tensor3(&[
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],
                [[0], [1i8]],
            ]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(-2i32);
    qp[3] = tensor1(&[1f32, 0.5]);
    suite.add(
        "scale_per_channel_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            kernel_format: OIHW,
            co: 2,
            group: 1,
            data: tensor2(&[[0i8]]),
            kernel: tensor3(&[[[0i8]], [[7]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[3] = tensor1(&[1f32, 0.5]);
    suite.add(
        "scale_per_channel_1",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            kernel_format: OIHW,
            co: 2,
            group: 1,
            data: tensor2(&[[0i8]]),
            kernel: tensor3(&[[[0i8]], [[7]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    suite.add(
        "a0_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[1i8]]),
            kernel: tensor3(&[[[0i8]]]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(-3i32);
    qp[2] = tensor0(2i32);
    suite.add(
        "a0_b0_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [2]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[0i8, 0]]),
            kernel: tensor3(&[[[0i8]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[2] = tensor0(1i32);
    suite.add(
        "kernel_zp",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            kernel_format: OIHW,
            co: 1,
            group: 1,
            data: tensor2(&[[1i8]]),
            kernel: tensor3(&[[[0i8]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[0] = tensor0(1i32);
    suite.add(
        "a0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            kernel_format: OIHW,
            co: 1,
            group: 1,
            data: tensor2(&[[0i8]]),
            kernel: tensor3(&[[[-1i8]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[2] = tensor0(1i32);
    suite.add(
        "b0_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [3]).unwrap(),
            kernel_format: OIHW,
            co: 1,
            group: 1,
            data: tensor2(&[[0i8, 0, 0]]),
            kernel: tensor3(&[[[0i8, 0]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[2] = tensor0(2i32);
    suite.add(
        "b0_1",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [6]).unwrap(),
            kernel_format: OIHW,
            co: 1,
            group: 1,
            data: tensor2(&[[0i8, 0, 0, 0, -20, 0]]),
            kernel: tensor3(&[[[0i8, 0]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    suite.add(
        "shape_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1, 2]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor3(&[[[0i8], [0]]]),
            kernel: tensor4(&[[[[0i8]]]]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    suite.add(
        "batch_0",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(3, 1, [2]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor3(&[[[0i8], [0]], [[0], [0]], [[0], [0]]]),
            kernel: tensor3(&[[[0i8, 0]]]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "batch_1",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(2, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[2, 1, 1]).unwrap(),
            kernel: tensor3(&[[[1i8]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
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
            data: tensor2(&[[-1i8]]),
            kernel: tensor3(&[[[1i8]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
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
            data: tensor2(&[[41i8]]),
            kernel: tensor3(&[[[1i8]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[1] = tensor0(0.5f32);
    suite.add(
        "scale_2",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[-1i8]]),
            kernel: tensor3(&[[[2i8]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[1] = tensor0(0.5f32);
    qp[5] = tensor0(2f32);
    suite.add(
        "scale_3",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[0i8]]),
            kernel: tensor3(&[[[0i8]]]),
            bias: Some(arr1(&[35i32])),
            qp,
            raw_output_dt: DatumType::I8,
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
            data: tensor2(&[[0i8]]),
            kernel: tensor3(&[[[0i8]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    suite.add(
        "group_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 2,
            data: tensor2(&[[0i8, 0]]),
            kernel: tensor3(&[[[0i8]], [[0]]]),
            bias: None,
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(1i32);
    suite.add(
        "group_1",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 2,
            data: tensor3(&[[[0i8], [0]]]),
            kernel: tensor3(&[[[1i8]], [[0]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(1i32);
    suite.add(
        "group_2",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 2,
            data: tensor2(&[[0i8, 0]]),
            kernel: tensor3(&[[[0i8]], [[1]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[0] = tensor0(2i32);
    qp[1] = tensor0(2f32);
    qp[3] = tensor0(0.5f32);
    qp[5] = tensor0(2f32);
    suite.add(
        "rounding_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: tensor2(&[[4i8]]),
            kernel: tensor3(&[[[-5i8]]]),
            bias: Some(arr1(&[-125i32])),
            qp,
            raw_output_dt: DatumType::I8,
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
            data: tensor2(&[[1i8]]),
            kernel: tensor3(&[[[0i8]], [[-15]]]),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    suite.add(
        "bias_1",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(1, 1, [1, 1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1, 1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 1, 1]).unwrap(),
            bias: Some(arr1(&[1, 2])),
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );

    let qp = qp_noop_i8();
    suite.add(
        "bias_2",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 1]).unwrap(),
            bias: Some(arr1(&[0, 1])),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[0] = tensor0(-1i32);
    let mut kernel = Tensor::zero::<i8>(&[5, 1, 2]).unwrap();
    *kernel.as_slice_mut::<i8>().unwrap().last_mut().unwrap() = -1;
    suite.add(
        "bias_3",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 1, [2]).unwrap(),
            co: 5,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[2, 1]).unwrap(),
            kernel,
            bias: Some(Array1::zeros([5])),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    suite.add(
        "bias_4",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(1, 1, [1, 1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1, 1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 1, 1]).unwrap(),
            bias: Some(arr1(&[0, 1])),
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );

    suite.add(
        "bias_5",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(1, 1, [1, 1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1, 1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[1, 1, 1, 1]).unwrap(),
            bias: Some(arr1(&[1])),
            qp: qp_noop_i8(),
            raw_output_dt: DatumType::I8,
        },
    );

    let qp = qp_noop_i8();
    suite.add(
        "bias_in_chw",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 1]).unwrap(),
            bias: Some(arr1(&[0, 0])),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "bias_with_batch",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[1, 1, 1]).unwrap(),
            bias: Some(arr1(&[1])),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "bias_vec_with_batch",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 1]).unwrap(),
            bias: Some(arr1(&[0, 1])),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "asan_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 2, [1]).unwrap(),
            co: 5,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 2]).unwrap(),
            kernel: Tensor::zero::<i8>(&[5, 2, 1]).unwrap(),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[3] = tensor1(&[1f32, 1f32]);
    suite.add(
        "tflite_per_axis_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 1]).unwrap(),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[3] = tensor1(&[1f32, 1f32]);
    suite.add(
        "tflite_per_axis_1",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1, 2]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1, 2]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 1, 2]).unwrap(),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[3] = tensor1(&[1f32, 1f32]);
    suite.add(
        "tflite_per_axis_nchw_0",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 1]).unwrap(),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[3] = tensor1(&[1f32, 1f32]);
    suite.add(
        "tflite_per_axis_nchw_1",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(1, 1, [2]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<i8>(&[1, 1, 2]).unwrap(),
            kernel: Tensor::zero::<i8>(&[2, 1, 2]).unwrap(),
            bias: None,
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "i8_u8",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            data: Tensor::zero::<u8>(&[1, 1]).unwrap(),
            kernel: Tensor::zero::<i8>(&[1, 1, 1]).unwrap(),
            bias: None,
            qp,
            raw_output_dt: DatumType::U8,
        },
    );
    /*
    let mut qp = qp_noop_i8();
    qp[2] = tensor0(-1i32);
    qp[5] = tensor0(0.5f32);
    suite.add(
    "i8_u8_0",
    QConvProblem {
    shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
    co: 1,
    kernel_format: OIHW,
    group: 1,
    data: Tensor::zero::<u8>(&[1, 1]).unwrap(),
    kernel: tensor3(&[[[1i8]]]),
    bias: None,
    qp,
    },
    );
    */
    let mut qp = qp_noop_i8();
    qp[3] = tensor0(2f32);
    suite.add(
        "i8_u8_ascale",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: Tensor::zero::<i8>(&[1, 1, 1]).unwrap(),
            bias: None,
            data: Tensor::zero::<u8>(&[1, 1]).unwrap(),
            qp,
            raw_output_dt: DatumType::U8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(1i32);
    suite.add(
        "i8_u8_d0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[-3i8]]]),
            bias: None,
            data: tensor2(&[[1u8]]),
            qp,
            raw_output_dt: DatumType::U8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[4] = tensor0(2i32);
    suite.add(
        "i8_u8_c0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[1i8]]]),
            bias: None,
            data: tensor2(&[[4u8]]),
            qp,
            raw_output_dt: DatumType::U8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "i8_u8_sat_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[-1i8]]]),
            bias: None,
            data: tensor2(&[[1u8]]),
            qp,
            raw_output_dt: DatumType::U8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(1i32);
    qp[4] = tensor0(2i32);
    suite.add(
        "i8_u8_weird_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[-1i8]]]),
            bias: None,
            data: tensor2(&[[0u8]]),
            qp,
            raw_output_dt: DatumType::U8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(1i32);
    qp[1] = tensor0(4f32);
    qp[3] = tensor0(2f32);
    suite.add(
        "i8_u8_scales_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[-1i8]]]),
            bias: None,
            data: tensor2(&[[0u8]]),
            qp,
            raw_output_dt: DatumType::U8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "u8_i8_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[0u8]]]),
            bias: None,
            data: tensor2(&[[0i8]]),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "u8_i8_1",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [2]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[0u8, 0]]]),
            bias: None,
            data: tensor2(&[[-9i8, 0]]),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let qp = qp_noop_i8();
    suite.add(
        "u8_i8_2",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[0u8]], [[0]]]),
            bias: None,
            data: tensor2(&[[0i8]]),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(6i32);
    suite.add(
        "u8_u8_i8_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[4u8]]]),
            bias: None,
            data: tensor2(&[[0i8]]),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(1i32);
    qp[1] = tensor0(2f32);
    qp[3] = tensor0(2f32);
    qp[4] = tensor0(2i32);
    suite.add(
        "many_qps_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[1u8]]]),
            bias: None,
            data: tensor2(&[[0u8]]),
            qp,
            raw_output_dt: DatumType::U8,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(1i32);
    suite.add(
        "i32_output_0",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[0i8]]]),
            bias: None,
            data: tensor2(&[[0i8]]),
            qp,
            raw_output_dt: DatumType::I32,
        },
    );
    let mut qp = qp_noop_i8();
    qp[1] = tensor0(0.25f32);
    qp[2] = tensor0(1i32);
    qp[5] = tensor0(0.5f32);
    suite.add(
        "i32_output_1",
        QConvProblem {
            shape_in: CHW.from_n_c_hw(1, 1, [1]).unwrap(),
            co: 1,
            kernel_format: OIHW,
            group: 1,
            kernel: tensor3(&[[[20i8]]]),
            bias: None,
            data: tensor2(&[[94i8]]),
            qp,
            raw_output_dt: DatumType::I32,
        },
    );
    let mut qp = qp_noop_i8();
    qp[0] = tensor0(-3);
    suite.add(
        "bin_by_scalar_and_bin_unicast_selection_0",
        QConvProblem {
            shape_in: NHWC.from_n_c_hw(2, 2, [4, 4]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 2,
            kernel: tensor4(&[[[[1i8]]], [[[0i8]]]]),
            bias: None,
            data: Tensor::zero::<i8>(&[2, 4, 4, 2]).unwrap(),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[3] = tensor1(&[1f32, 1f32]);
    suite.add(
        "batch_vec_scale",
        QConvProblem {
            shape_in: NCHW.from_n_c_hw(2, 1, [2]).unwrap(),
            co: 2,
            kernel_format: OIHW,
            group: 1,
            kernel: Tensor::zero::<i8>(&[2, 1, 1]).unwrap(),
            bias: None,
            data: Tensor::zero::<i8>(&[2, 1, 2]).unwrap(),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );

    let mut qp = qp_noop_i8();
    qp[1] = tensor1(&[0.5f32]);
    qp[3] = tensor1(&[1f32; 18]);
    qp[5] = tensor1(&[0.5f32]);
    suite.add(
        "timeout_0",
        QConvProblem {
            shape_in: HWC.from_n_c_hw(1, 3, [1]).unwrap(),
            co: 18,
            kernel_format: OIHW,
            group: 3,
            kernel: Tensor::zero::<i8>(&[18, 1, 1]).unwrap(),
            bias: None,
            data: Tensor::zero::<i8>(&[1, 3]).unwrap(),
            qp,
            raw_output_dt: DatumType::I8,
        },
    );
    Ok(suite)
}
