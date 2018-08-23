#[macro_use]
extern crate criterion;
extern crate ndarray;
#[macro_use]
extern crate tfdeploy;

use criterion::Criterion;

use tfdeploy::ops::nn::conv2d::*;
use tfdeploy::ops::nn::local_patch::*;
use tfdeploy::*;

use tfdeploy::ops::Op;

fn mk(sizes: &[usize]) -> Tensor {
    let data = ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
        .into_shape(sizes)
        .unwrap();
    Tensor::F32(data)
}

fn conv(bencher: &mut Criterion) {
    let stride = 1;
    let conv = Conv2D::<f32>::new(LocalPatch::valid(stride, stride));
    let inputs = tvec![mk(&[1, 82, 1, 40]).into(), mk(&[41, 1, 40, 128]).into()];
    conv.eval(inputs.clone()).unwrap();
    bencher.bench_function("Conv2D<f32>(1x82x1x40 41x1x40x128)", move |b| {
        b.iter(|| conv.eval(inputs.clone()).unwrap())
    });
}

criterion_group!(benches, conv);
criterion_main!(benches);
