#![allow(dead_code)]
extern crate criterion;
#[macro_use]
extern crate derive_new;
extern crate ndarray;
extern crate tract_core;
use criterion::*;

use tract_core::model::*;
use tract_core::*;

use nn::DataFormat::{CHW, HWC};
use tract_core::internal::*;
use tract_core::ops::{cnn, nn};

#[derive(Debug, new)]
struct Problem {
    input: nn::DataShape,
    kernel_geo: TVec<usize>,
    co: usize,
    strides: TVec<usize>,
    dil: TVec<usize>,
}

impl Problem {
    pub fn image(&self) -> Tensor {
        Tensor::from(ndarray::ArrayD::<f32>::zeros(&*self.input.shape))
    }

    pub fn image_fact(&self) -> TypedFact {
        TypedFact::dt_shape(DatumType::F32, &*self.input.shape).unwrap()
    }

    pub fn image_type(&self) -> TypedFact {
        TypedFact::dt_shape(f32::datum_type(), &*self.input.shape).unwrap()
    }

    pub fn to_plan(&self, direct: bool) -> SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel> {
        assert_eq!(self.input.hw_rank(), self.kernel_geo.len());
        assert_eq!(self.input.hw_rank(), self.dil.len());
        assert_eq!(self.input.hw_rank(), self.strides.len());

        let mut full_kernel_shape = self.kernel_geo.clone();
        full_kernel_shape.push(*self.input.c());
        full_kernel_shape.push(self.co);
        let kernel = Tensor::zero::<f32>(&*full_kernel_shape).unwrap();
        let conv = cnn::ConvUnary {
            pool_spec: cnn::PoolSpec {
                data_format: self.input.fmt,
                kernel_shape: self.kernel_geo.clone(),
                padding: cnn::PaddingSpec::Valid,
                dilations: Some(self.dil.clone()),
                strides: Some(self.strides.clone()),
                output_channel_override: Some(self.co),
            },
            kernel_fmt: cnn::KernelFormat::HWIO,
            kernel: kernel.into_arc_tensor(),
            group: 1,
            bias: None,
            q_params: None,
        };

        let mut model = TypedModel::default();
        let input = model.add_source("input", self.image_type()).unwrap();
        let output = unsafe { conv.wire_as_im2col_pair(&mut model, "", input, direct).unwrap() };
        model.set_output_outlets(&[output]).unwrap();
        SimplePlan::new(model).unwrap()
    }

    pub fn bench(&self, b: &mut Bencher, direct: bool) {
        let image = self.image();
        let im2col_plan = self.to_plan(direct);
        let args = tvec!(image.clone().into());
        b.iter(|| im2col_plan.run(args.clone()).unwrap())
    }
}

fn b(c: &mut Criterion, name: &str, pbs: Vec<(usize, Problem)>) {
    let mut group = c.benchmark_group(name);
    for (i, pb) in pbs {
        let image = pb.image();
        let direct_plan = pb.to_plan(false);
        let args = tvec!(image.clone().into());
        let output = direct_plan.run(args.clone()).unwrap();
        let len = output[0].len();
        let tp = Throughput::Elements(
            (len * pb.input.c() * pb.kernel_geo.iter().product::<usize>()) as _,
        );
        group.throughput(tp);
        group.bench_with_input(BenchmarkId::new("im2col", i), &pb, |b, pb| pb.bench(b, false));
        group.bench_with_input(BenchmarkId::new("direct", i), &pb, |b, pb| pb.bench(b, true));
    }
}

fn size(c: &mut Criterion) {
    let pbs = [16, 32, 64, 128]
        .iter()
        .map(|&s| {
            (
                s,
                Problem::new(
                    HWC.from_n_c_hw(1, 32, &[s, s]).unwrap(),
                    tvec!(3, 3),
                    32,
                    tvec!(1, 1),
                    tvec!(1, 1),
                ),
            )
        })
        .collect();
    b(c, "size", pbs);
}

fn kernel_sq(c: &mut Criterion) {
    let pbs = [1, 2, 3, 4, 5]
        .iter()
        .map(|&s| {
            (
                s,
                Problem::new(
                    HWC.from_n_c_hw(1, 32, &[64, 64]).unwrap(),
                    tvec!(s, s),
                    32,
                    tvec!(1, 1),
                    tvec!(1, 1),
                ),
            )
        })
        .collect();
    b(c, "kernel_sq", pbs);
}

fn kernel_1d(c: &mut Criterion) {
    let pbs = [1, 3, 5, 8, 10, 15, 20, 30, 40]
        .iter()
        .map(|&s| {
            (
                s,
                Problem::new(
                    HWC.from_n_c_hw(1, 32, &[64]).unwrap(),
                    tvec!(s),
                    32,
                    tvec!(1),
                    tvec!(1),
                ),
            )
        })
        .collect();
    b(c, "kernel_1d", pbs);
}

fn ci(c: &mut Criterion) {
    let pbs = [1, 2, 4, 8, 16, 32, 64]
        .iter()
        .map(|&s| {
            (
                s,
                Problem::new(
                    HWC.from_n_c_hw(1, s, &[64, 64]).unwrap(),
                    tvec!(3, 3),
                    32,
                    tvec!(1, 1),
                    tvec!(1, 1),
                ),
            )
        })
        .collect();
    b(c, "ci", pbs);
}

fn co(c: &mut Criterion) {
    let pbs = [1, 2, 4, 8, 16, 32, 64]
        .iter()
        .map(|&s| {
            (
                s,
                Problem::new(
                    HWC.from_n_c_hw(1, 32, &[64, 64]).unwrap(),
                    tvec!(3, 3),
                    s,
                    tvec!(1, 1),
                    tvec!(1, 1),
                ),
            )
        })
        .collect();
    b(c, "co", pbs);
}

#[rustfmt::skip]
mod b {
    use super::*;
    macro_rules! b {
        ($id:ident, $($args:expr),*) => {
            #[allow(non_snake_case)]
            pub fn $id(c: &mut Criterion) {
                b(c, stringify!($id), vec!((1, Problem::new($($args),*))));
            }
        }
    }

    b!(ARM_ML_KWS_CNN_M_0, HWC.from_n_c_hw(1, 1,  &[49, 10]).unwrap(),   tvec!(10, 4), 64, tvec!(1, 1), tvec!(1, 1));
    b!(ARM_ML_KWS_CNN_M_1, HWC.from_n_c_hw(1, 64, &[40, 7]).unwrap(),    tvec!(10, 4), 48, tvec!(2, 1), tvec!(1, 1));
    b!(Hey_Snips_v4_dil1,  HWC.from_n_c_hw(1, 16, &[10]).unwrap(),       tvec!(3),     64, tvec!(1),    tvec!(1));
    b!(Hey_Snips_v4_dil2,  HWC.from_n_c_hw(1, 16, &[12]).unwrap(),       tvec!(3),     64, tvec!(1),    tvec!(2));
    b!(Hey_Snips_v4_dil4,  HWC.from_n_c_hw(1, 16, &[16]).unwrap(),       tvec!(3),     64, tvec!(1),    tvec!(4));
    b!(Hey_Snips_v4_dil8,  HWC.from_n_c_hw(1, 16, &[24]).unwrap(),       tvec!(3),     64, tvec!(1),    tvec!(8));
    b!(Conv2d_2a_3x3,      HWC.from_n_c_hw(1, 32, &[149, 149]).unwrap(), tvec!(3, 3),  32, tvec!(1, 1), tvec!(1, 1));
}

criterion_group!(
    benches,
    b::ARM_ML_KWS_CNN_M_0,
    b::ARM_ML_KWS_CNN_M_1,
    b::Hey_Snips_v4_dil1,
    b::Hey_Snips_v4_dil2,
    b::Hey_Snips_v4_dil4,
    b::Hey_Snips_v4_dil8,
    b::Conv2d_2a_3x3,
    size,
    kernel_sq,
    kernel_1d,
    ci,
    co,
);
criterion_main!(benches);
