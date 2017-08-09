#[macro_use]
extern crate bencher;
extern crate ndarray;
extern crate tfdeploy;

use tfdeploy::*;
use tfdeploy::ops::conv::*;
use tfdeploy::ops::Op;

fn mk(sizes: &[usize]) -> Matrix {
    let data = ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
        .into_shape(sizes)
        .unwrap();
    Matrix::F32(data)
}

fn bench(bencher: &mut bencher::Bencher) {
    let stride = 1;
    let strides = vec![1, stride, stride, 1];
    let conv = Conv2D {
        padding: Padding::Valid,
        strides: strides,
        _data_format: DataFormat::NHWC,
    };
    let inputs = vec![mk(&[1, 82, 1, 40]), mk(&[41, 1, 40, 128])];
    conv.eval(inputs.clone()).unwrap();
    bencher.iter(|| conv.eval(inputs.clone()).unwrap())
}

benchmark_group!(benches, bench);
benchmark_main!(benches);
