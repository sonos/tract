use crate::internal::*;
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
    data_format: DataFormat,
    kernel_format: KernelFormat,
    input: ArrayD<f32>,
    kernel: ArrayD<f32>,
    strides: TVec<usize>,
    dilations: TVec<usize>,
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
        (1usize..4)
            .prop_flat_map(|georank| {
                (
                    any::<DataFormat>(),
                    any::<KernelFormat>(),
                    1usize..3,
                    1usize..4,
                    1usize..4,
                    vec(1usize..4, georank..=georank), // kernel shape
                    vec(1usize..8, georank..=georank), // image shape
                    vec(1usize..4, georank..=georank), // strides
                    vec(1usize..4, georank..=georank), // dilations
                )
            })
            .prop_flat_map(|(df, kf, n, ci, co, hwk, hwi, strides, dilations)| {
                let mut kernel_shape = hwk;
                match kf {
                    OIHW => {
                        kernel_shape.insert(0, ci);
                        kernel_shape.insert(1, co);
                    }
                    HWIO => {
                        kernel_shape.push(co);
                        kernel_shape.push(ci);
                    }
                };
                let data_shape = df.from_n_c_hw(n, ci, &hwi).unwrap();
                (
                    Just(df),
                    Just(kf),
                    tensor(&data_shape.shape),
                    tensor(&kernel_shape),
                    Just(strides),
                    Just(dilations),
                )
            })
            .prop_map(|(data_format, kernel_format, input, kernel, strides, dilations)| DeconvProblem {
                data_format,
                kernel_format,
                input,
                kernel,
                strides: strides.into(),
                dilations: dilations.into(),
            })
            .boxed()
    }
}

impl DeconvProblem {
    fn tract(&self) -> ArrayD<f32> {
        let op = DeconvUnary::new(
            self.data_format,
            self.kernel_format,
            PaddingSpec::Valid,
            self.kernel.clone().into_arc_tensor(),
            self.strides.clone(),
            self.dilations.clone(),
        );
        let mut outputs = op.eval(tvec!(self.input.clone().into_arc_tensor())).unwrap();
        outputs.remove(0).into_tensor().into_array().unwrap().into_dimensionality().unwrap()
    }

    fn reference(&self) -> ArrayD<f32> {
        use std::iter::once;
        let co = match self.kernel_format {
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 2],
            KernelFormat::OIHW => self.kernel.shape()[1],
        };
        let input_shape = self.data_format.shape(self.input.shape()).unwrap();
        let n = if self.data_format.has_n() { self.input.shape()[0] } else { 1 };
        let kernel_hwdims = self.kernel_format.spatial_shape(self.kernel.shape());
        let output_shape_geo: TVec<usize> =
            tract_itertools::izip!(input_shape.hw_dims(), kernel_hwdims, self.strides.iter(), self.dilations.iter())
                .map(|(i, k, s, d)| (i - 1) * s + (k - 1) * d + 1)
                .collect();
        let output_shape = self.data_format.from_n_c_hw(n, co, output_shape_geo).unwrap();
        let mut output = ArrayD::zeros(&*output_shape.shape);
        for n in 0..n {
            for co in 0..co {
                for ci in 0..*input_shape.c() {
                    for hwi in indices(input_shape.hw_dims()) {
                        for hwk in indices(kernel_hwdims) {
                            let hwo: TVec<usize> = tract_itertools::izip!(
                                hwi.slice().iter(),
                                hwk.slice().iter(),
                                self.strides.iter(),
                                self.dilations.iter(),
                            )
                            .map(|(i, k, s, d)| i * s + k * d)
                            .collect();
                            let i = self.data_format.from_n_c_hw(n, ci, hwi.slice()).unwrap();
                            let o = self.data_format.from_n_c_hw(n, co, hwo).unwrap();
                            let k: TVec<usize> = match self.kernel_format {
                                OIHW => once(ci)
                                    .chain(once(co))
                                    .chain(hwk.slice().iter().cloned())
                                    .collect(),
                                HWIO => hwk
                                    .slice()
                                    .iter()
                                    .cloned()
                                    .chain(once(co))
                                    .chain(once(ci))
                                    .collect(),
                            };
                            output[&*o.shape] += self.input[&*i.shape] * self.kernel[&*k];
                        }
                    }
                }
            }
        }
        output
    }
}

proptest::proptest! {
    #[test]
    fn prop(pb in any::<DeconvProblem>()) {
        prop_assert_eq!(pb.tract(), pb.reference());
    }
}

#[test]
fn test_trivial_0() {
    let pb = DeconvProblem {
        data_format: NCHW,
        kernel_format: OIHW,
        input: arr4(&[[[[0.0]]]]).into_dyn(),
        kernel: arr4(&[[[[0.0]]]]).into_dyn(),
        strides: tvec!(1, 1),
        dilations: tvec!(1, 1),
    };
    assert_eq!(pb.tract(), pb.reference());
}

#[test]
fn test_hwc_0() {
    let pb = DeconvProblem {
        data_format: HWC,
        kernel_format: OIHW,
        input: arr3(&[[[0.0]], [[0.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0]]]]).into_dyn(),
        strides: tvec!(1, 1),
        dilations: tvec!(1, 1),
    };
    assert_eq!(pb.tract(), pb.reference());
}

#[test]
fn test_geo_0() {
    let pb = DeconvProblem {
        data_format: HWC,
        kernel_format: OIHW,
        input: arr3(&[[[0.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0], [0.0]]]]).into_dyn(),
        strides: tvec!(1, 1),
        dilations: tvec!(1, 1),
    };
    assert_eq!(pb.tract(), pb.reference());
}

#[test]
fn test_hwio_0() {
    let pb = DeconvProblem {
        data_format: HWC,
        kernel_format: HWIO,
        input: arr3(&[[[0.0]]]).into_dyn(),
        kernel: arr4(&[[[[0.0], [0.0]]]]).into_dyn(),
        strides: tvec!(1, 1),
        dilations: tvec!(1, 1),
    };
    assert_eq!(pb.tract(), pb.reference());
}

#[test]
fn test_strides_1() {
    let pb = DeconvProblem {
        data_format: HWC,
        kernel_format: OIHW,
        input: arr2(&[[0.0], [1.0]]).into_dyn(),
        kernel: arr3(&[[[1.0]]]).into_dyn(),
        strides: tvec!(2),
        dilations: tvec!(1),
    };
    assert_eq!(pb.tract(), pb.reference());
}
