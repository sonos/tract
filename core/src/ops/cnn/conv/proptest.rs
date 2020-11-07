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
    data: ArrayD<f32>,
    kernel: ArrayD<f32>,
}

impl ConvProblem {
    fn geo_ker(&self) -> &[usize] {
        &self.kernel.shape()[self.kernel_format.h_axis()..][..self.shape_in.hw_rank()]
    }

    fn reference(&self) -> ArrayD<f32> {
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        let mut out = ArrayD::zeros(&*self.shape_out.shape);
        let n = *self.shape_in.n().clone().unwrap_or(&1);
        for n in 0..n {
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
                    for ci in 0..*self.shape_in.c() {
                        input_coords[self.shape_in.c_axis()] = ci;
                        let i = self.data[&*input_coords];
                        eprintln!("I: {:?} {}", input_coords, i);
                        for co in 0..*self.shape_out.c() {
                            output_coords[self.shape_out.c_axis()] = co;
                            let mut kernel_coords: TVec<usize> = geo_ker.slice().into();
                            match self.kernel_format {
                                KernelFormat::OIHW => {
                                    kernel_coords.insert(0, ci);
                                    kernel_coords.insert(0, co);
                                }
                                KernelFormat::HWIO => {
                                    kernel_coords.push(ci);
                                    kernel_coords.push(co);
                                }
                            }
                            let k = self.kernel[&*kernel_coords];
                            eprintln!("K: {:?} {}", kernel_coords, k);
                            out[&*output_coords] += k * i;
                        }
                    }
                }
            }
        }
        out
    }

    fn tract(&self) -> anyhow::Result<ArrayD<f32>> {
        assert_eq!(self.data.shape(), &*self.shape_in.shape);
        let mut model = TypedModel::default();
        let wire = model
            .add_source("input", TypedFact::dt_shape(f32::datum_type(), &*self.shape_in.shape)?)?;
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
            1,
            None,
            None,
        );
        let wire = model.wire_node("conv", op, &[wire])?[0];
        model.set_output_outlets(&[wire])?;
        dbg!(&model);
        let mut output =
            model.into_optimized()?.into_runnable()?.run(tvec![self.data.clone().into_tensor()])?;
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
            1usize..3,
            1usize..4,
            1usize..4,
            (1usize..3).prop_flat_map(|r| shapes(r)),
        )
            .prop_flat_map(|(df, kf, n, ci, co, (mut ker_shape, data_shape))| {
                dbg!(&ker_shape);
                dbg!(&data_shape);
                let shape_in = df.from_n_c_hw(n, ci, &data_shape).unwrap();
                let shape_out: TVec<_> =
                    izip!(&ker_shape, data_shape).map(|(k, d)| d - k + 1).collect();
                let shape_out = df.from_n_c_hw(n, co, &shape_out).unwrap();
                let data_in = tensor(shape_in.shape.iter().cloned().collect());
                match kf {
                    KernelFormat::HWIO => {
                        ker_shape.push(ci);
                        ker_shape.push(co)
                    }
                    KernelFormat::OIHW => {
                        ker_shape.insert(0, ci);
                        ker_shape.insert(0, co)
                    }
                };
                let kernel = tensor(ker_shape);
                (Just((kf, shape_in, shape_out)), data_in, kernel)
            })
            .prop_map(|((kernel_format, shape_in, shape_out), data, kernel)| ConvProblem {
                shape_in,
                shape_out,
                kernel_format,
                data,
                kernel,
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
        data: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
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
        data: ndarray::arr3(&[[[1.0f32], [0.0]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[1.0f32]]]).into_dyn(),
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
        data: ndarray::arr3(&[[[0.0f32, 1.0]]]).into_dyn(),
        kernel: ndarray::arr3(&[[[0.0f32], [1.0]]]).into_dyn(),
    };
    assert_eq!(pb.tract()?, pb.reference());
    Ok(())
}
