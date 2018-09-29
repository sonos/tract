#[macro_use]
extern crate criterion;
extern crate ndarray;
#[macro_use]
extern crate tfdeploy;
extern crate tfdeploy_tf;

use criterion::Criterion;

use tfdeploy::ops::nn::{Conv, PaddingSpec};
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

fn tfd_conv(bencher: &mut Criterion) {
    let conv = Conv::new(true, true, None, None, PaddingSpec::Valid, None);
    let inputs = tvec![mk(&[1, 82, 1, 40]).into(), mk(&[41, 1, 40, 128]).into()];
    conv.eval(inputs.clone()).unwrap();
    bencher.bench_function("tfd::Conv<f32>(1x82x1x40 41x1x40x128)", move |b| {
        b.iter(|| conv.eval(inputs.clone()).unwrap())
    });
}

criterion_group!(benches, tfd_tf_conv2d, tfd_conv);
criterion_main!(benches);
