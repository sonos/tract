#[macro_use]
extern crate criterion;
extern crate ndarray;
#[macro_use]
extern crate tfdeploy;
extern crate tfdeploy_tf;

use criterion::Criterion;

use tfdeploy::ops::Value;
use tfdeploy::ops::nn::{Conv, FixedParamsConv, PaddingSpec};
use tfdeploy_tf::ops::nn::conv2d::*;
use tfdeploy_tf::ops::nn::local_patch::*;
use tfdeploy::*;

use tfdeploy::ops::Op;

fn mk(sizes: &[usize]) -> Tensor {
    let data = ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
        .into_shape(sizes)
        .unwrap();
    Tensor::F32(data)
}

fn tfd_tf_conv2d(bencher: &mut Criterion) {
    let conv = Conv2D::<f32>::new(LocalPatch::valid(1, 1));
    let inputs = tvec![mk(&[1, 82, 1, 40]).into(), mk(&[41, 1, 40, 128]).into()];
    conv.eval(inputs.clone()).unwrap();
    bencher.bench_function("tfd_td::Conv2D<f32>(1x82x1x40 41x1x40x128)", move |b| {
        b.iter(|| conv.eval(inputs.clone()).unwrap())
    });
}

fn tfd_conv_gen(bencher: &mut Criterion) {
    let conv = Conv::new(true, true, None, None, PaddingSpec::Valid, None);
    let inputs = tvec![mk(&[1, 82, 1, 40]).into(), mk(&[41, 1, 40, 128]).into()];
    conv.eval(inputs.clone()).unwrap();
    bencher.bench_function("tfd::Conv<f32>(1x82x1x40 41x1x40x128)", move |b| {
        b.iter(|| conv.eval(inputs.clone()).unwrap())
    });
}

fn tfd_conv_fixed(bencher: &mut Criterion) {
    let conv = Conv::new(true, true, None, None, PaddingSpec::Valid, None);
    let input:TVec<Value> = tvec![mk(&[1, 82, 1, 40]).into()];
    let kernel = mk(&[41, 1, 40, 128]);
    let optim = FixedParamsConv::new(&conv, input[0].shape(), kernel.to_array_view::<f32>().unwrap(), None).unwrap();
    optim.eval(input.clone()).unwrap();
    bencher.bench_function("tfd::Convoler<f32>(1x82x1x40 41x1x40x128)", move |b| {
        b.iter(|| optim.eval(input.clone()).unwrap())
    });
}

criterion_group!(benches, tfd_tf_conv2d, tfd_conv_gen, tfd_conv_fixed);
criterion_main!(benches);
