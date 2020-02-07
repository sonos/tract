#![allow(dead_code)]
#[macro_use]
extern crate criterion;
#[macro_use]
extern crate derive_new;
extern crate ndarray;
extern crate tract_core;
use criterion::Criterion;

use tract_core::model::*;
use tract_core::*;

use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ops::cnn::ConvUnary;

use std::convert::TryInto;

#[derive(Debug, new)]
struct Problem {
    h: usize,
    w: usize,
    ci: usize,
    kh: usize,
    kw: usize,
    co: usize,
    stride_h: usize,
    stride_w: usize,
    dil_h: usize,
    dil_w: usize,
}

impl Default for Problem {
    fn default() -> Problem {
        Problem {
            h: 64,
            w: 64,
            kh: 3,
            kw: 3,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
            ci: 32,
            co: 32,
        }
    }
}

impl Problem {
    pub fn image(&self) -> Tensor {
        Tensor::from(ndarray::ArrayD::<f32>::zeros(&*self.image_shape()))
    }
    pub fn image_shape(&self) -> TVec<usize> {
        tvec!(1, self.h, self.w, self.ci)
    }

    pub fn image_fact(&self) -> InferenceFact {
        InferenceFact::dt_shape(DatumType::F32, self.image_shape())
    }

    pub fn image_type(&self) -> TypedFact {
        TypedFact::dt_shape(f32::datum_type(), &*self.image_shape()).unwrap()
    }

    pub fn to_unary(&self) -> Box<ConvUnary> {
        let kernel =
            Tensor::from(ndarray::Array4::<f32>::zeros((self.kh, self.kw, self.ci, self.co)));
        let conv = tract_core::ops::cnn::Conv::default()
            .nhwc()
            .hwio()
            .dilations(tvec!(self.dil_h, self.dil_w))
            .kernel_shape(kernel.shape()[0..2].into())
            .strides(tvec!(self.stride_h, self.stride_w));
        let kernel_fact: TypedFact = TypedFact::from(kernel);
        let image_fact: TypedFact = (&self.image_fact()).try_into().unwrap();
        let unary = conv.to_unary(&[&image_fact, &kernel_fact]).unwrap();
        Box::new(unary.unwrap())
    }

    pub fn to_plan(&self, direct: bool) -> SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel> {
        let unary = self.to_unary();
        let mut model_im2col = TypedModel::default();
        let input = model_im2col.add_source("input", self.image_type()).unwrap();
        let output =
            unsafe { unary.wire_as_im2col_pair(&mut model_im2col, "", input, direct).unwrap() };
        model_im2col.set_output_outlets(&[output]).unwrap();
        SimplePlan::new(model_im2col).unwrap()
    }
}

fn b(c: &mut Criterion, name: &str, pbs: Vec<Problem>) {
    c.bench(
        name,
        criterion::ParameterizedBenchmark::new(
            "im2col",
            move |b, pb| {
                let image = pb.image();
                let im2col_plan = pb.to_plan(false);
                let args = tvec!(image.clone().into());
                b.iter(|| im2col_plan.run(args.clone()).unwrap())
            },
            pbs,
        )
        .with_function("direct", move |b, pb| {
            let image = pb.image();
            let direct_plan = pb.to_plan(true);
            let args = tvec!(image.clone().into());
            b.iter(|| direct_plan.run(args.clone()).unwrap())
        })
        .throughput(|pb| {
            let h = (pb.h - (pb.kh - 1) * pb.dil_h + 1) / pb.stride_h;
            let w = (pb.w - (pb.kw - 1) * pb.dil_w + 1) / pb.stride_w;
            criterion::Throughput::Elements((h * w * pb.ci * pb.co * pb.kh * pb.kw) as _)
        }),
    );
}

fn size(c: &mut Criterion) {
    let pbs =
        [16, 32, 64, 128].iter().map(|&s| Problem { h: s, w: s, ..Problem::default() }).collect();
    b(c, "size", pbs);
}

fn kernel_sq(c: &mut Criterion) {
    let pbs =
        [1, 2, 3, 4, 5].iter().map(|&s| Problem { kh: s, kw: s, ..Problem::default() }).collect();
    b(c, "kernel_sq", pbs);
}

fn kernel_1d(c: &mut Criterion) {
    let pbs = [1, 3, 5, 8, 10, 15, 20, 30, 40]
        .iter()
        .map(|&s| Problem { kh: s, kw: 1, ..Problem::default() })
        .collect();
    b(c, "kernel_1d", pbs);
}

fn ci(c: &mut Criterion) {
    let pbs =
        [1, 2, 4, 8, 16, 32, 64].iter().map(|&s| Problem { ci: s, ..Problem::default() }).collect();
    b(c, "ci", pbs);
}

fn co(c: &mut Criterion) {
    let pbs =
        [1, 2, 4, 8, 16, 32, 64].iter().map(|&s| Problem { co: s, ..Problem::default() }).collect();
    b(c, "co", pbs);
}

macro_rules! b {
    ($id:ident, $($args:expr),*) => {
        #[allow(non_snake_case)]
        fn $id(c: &mut Criterion) {
            b(c, stringify!($id), vec!(Problem::new($($args),*)));
        }
    }
}

b!(ARM_ML_KWS_CNN_M_0, 49, 10, 1, 10, 4, 64, 1, 1, 1, 1);
b!(ARM_ML_KWS_CNN_M_1, 40, 7, 64, 10, 4, 48, 2, 1, 1, 1);
b!(Hey_Snips_v4_dil1, 10, 16, 1, 3, 1, 64, 1, 1, 1, 1);
b!(Hey_Snips_v4_dil2, 12, 16, 1, 3, 1, 64, 1, 1, 2, 1);
b!(Hey_Snips_v4_dil4, 16, 16, 1, 3, 1, 64, 1, 1, 4, 1);
b!(Hey_Snips_v4_dil8, 24, 16, 1, 3, 1, 64, 1, 1, 8, 1);
b!(Conv2d_2a_3x3, 149, 149, 32, 3, 3, 32, 1, 1, 1, 1);

criterion_group!(
    benches,
    ARM_ML_KWS_CNN_M_0,
    ARM_ML_KWS_CNN_M_1,
    Hey_Snips_v4_dil1,
    Hey_Snips_v4_dil2,
    Hey_Snips_v4_dil4,
    Hey_Snips_v4_dil8,
    //    Conv2d_2a_3x3,
    size,
    kernel_sq,
    kernel_1d,
    ci,
    co,
);
criterion_main!(benches);
