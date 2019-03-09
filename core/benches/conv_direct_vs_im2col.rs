#[macro_use]
extern crate criterion;
extern crate ndarray;
extern crate tract_core;
use criterion::Criterion;

use tract_core::model::ModelDsl;
use tract_core::*;

use tract_core::ops::nn::ConvUnary;
use tract_core::ops::nn::PaddingSpec;
use tract_core::ops::nn::PaddingSpec::SameUpper as Same;
use tract_core::ops::nn::PaddingSpec::Valid;
use tract_core::ops::prelude::*;

fn b(
    c: &mut Criterion,
    name: &str,
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
    padding: PaddingSpec,
) {
    let image = Tensor::from(ndarray::Array4::<f32>::zeros((1, h, w, ci)));
    let kernel = Tensor::from(ndarray::Array4::<f32>::zeros((kh, kw, ci, co)));
    let conv = tract_core::ops::nn::Conv::new(
        tract_core::ops::nn::DataFormat::NHWC,
        tract_core::ops::nn::KernelFormat::HWIO,
        Some(tvec!(dil_h, dil_w)),
        Some(kernel.shape()[0..2].into()),
        padding,
        Some(tvec!(stride_h, stride_w)),
        1,
    );
    let input_fact = TensorFact::dt_shape(DatumType::F32, image.shape());
    let kernel_fact = TensorFact::from(kernel);
    let unary = conv
        .reduce(tvec!(&input_fact, &kernel_fact), tvec!(), ReductionPhase::Normalize)
        .unwrap()
        .unwrap()
        .ops
        .remove(0);
    let unary = unary.downcast_ref::<ConvUnary>().unwrap();

    let (im2col, cvgemm) = unary.to_boxed_im2col_pair::<f32>(&*image.shape()).unwrap();
    let mut model_im2col = Model::default();
    model_im2col.add_source("input").unwrap();
    model_im2col.chain("im2col", im2col).unwrap();
    model_im2col.chain("gemm", cvgemm).unwrap();
    let im2col_plan = SimplePlan::new(model_im2col).unwrap();

    let direct = unary.to_direct(&*image.shape()).unwrap();
    let mut model_direct = Model::default();
    model_direct.add_source("input").unwrap();
    model_direct.chain("conv", Box::new(direct)).unwrap();
    let direct_plan = SimplePlan::new(model_direct).unwrap();

    let args = tvec!(image.clone().into());
    let r_im2col = im2col_plan.run(args.clone()).unwrap();
    let r_direct = direct_plan.run(args.clone()).unwrap();
    assert!(r_im2col[0].close_enough(&*r_direct[0], true));

    let args_2 = tvec!(image.into());
    c.bench(
        name,
        criterion::ParameterizedBenchmark::new(
            "im2col",
            move |b, _| b.iter(|| im2col_plan.run(args.clone()).unwrap()),
            vec![()],
        )
        .with_function("direct", move |b, _| b.iter(|| direct_plan.run(args_2.clone()).unwrap())),
    );
}

macro_rules! b {
    ($id:ident, $($args:expr),*) => {
        #[allow(non_snake_case)]
        fn $id(c: &mut Criterion) {
            b(c, stringify!($id), $($args),*);
        }
    }
}

b!(ARM_ML_KWS_CNN_M_0, 49, 10, 1, 10, 4, 64, 1, 1, 1, 1, Valid);
b!(ARM_ML_KWS_CNN_M_1, 40, 7, 64, 10, 4, 48, 2, 1, 1, 1, Valid);
b!(Hey_Snips_v4_dil1, 10, 16, 1, 3, 1, 64, 1, 1, 1, 1, Valid);
b!(Hey_Snips_v4_dil2, 12, 16, 1, 3, 1, 64, 1, 1, 2, 1, Valid);
b!(Hey_Snips_v4_dil4, 16, 16, 1, 3, 1, 64, 1, 1, 4, 1, Valid);
b!(Hey_Snips_v4_dil8, 24, 16, 1, 3, 1, 64, 1, 1, 8, 1, Valid);
b!(Conv2d_2a_3x3, 149, 149, 32, 3, 3, 32, 1, 1, 1, 1, Valid);
b!(Conv2d_2b_3x3, 147, 147, 32, 3, 3, 64, 1, 1, 1, 1, Same);

criterion_group!(
    benches,
    ARM_ML_KWS_CNN_M_0,
    ARM_ML_KWS_CNN_M_1,
    Hey_Snips_v4_dil1,
    Hey_Snips_v4_dil2,
    Hey_Snips_v4_dil4,
    Hey_Snips_v4_dil8,
    Conv2d_2a_3x3,
    Conv2d_2b_3x3,
);
criterion_main!(benches);
