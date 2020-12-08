#[macro_use]
extern crate criterion;
extern crate ndarray;
extern crate tract_core;
use criterion::Criterion;

use tract_core::internal::*;
use tract_core::ops::cnn::conv::Im2Col;
use tract_core::ops::cnn::PaddingSpec;
use tract_core::ops::cnn::PaddingSpec::SameUpper as Same;
use tract_core::ops::cnn::PaddingSpec::Valid;
use tract_core::ops::{cnn, nn};

fn b(
    c: &mut Criterion,
    name: &str,
    h: usize,
    w: usize,
    ci: usize,
    kh: usize,
    kw: usize,
    co: usize,
    stride: usize,
    padding: PaddingSpec,
) {
    let image = Tensor::from(ndarray::Array4::<f32>::zeros((1, h, w, ci)));
    let kernel = Tensor::from(ndarray::Array4::<f32>::zeros((kh, kw, ci, co))).into_arc_tensor();
    let unary = cnn::ConvUnary {
        pool_spec: cnn::PoolSpec {
            data_format: nn::DataFormat::NHWC,
            kernel_shape: tvec!(kh, kw),
            padding,
            dilations: None,
            strides: Some(tvec!(stride, stride)),
            output_channel_override: None,
        },
        kernel_fmt: cnn::KernelFormat::HWIO,
        kernel,
        group: 1,
        bias: None,
        quantized: false,
    };

    let mut m = TypedModel::default();
    let wire = m.add_source("", TypedFact::dt_shape(f32::datum_type(), &[1, h, w, ci])).unwrap();
    unsafe {
        unary.wire_as_im2col_pair(&mut m, "", wire).unwrap();
    }
    let im2col = m.node(1).op_as::<Im2Col>().unwrap();
    let args = tvec!(image.into());
    c.bench_function(name, move |b| b.iter(|| im2col.eval(args.clone()).unwrap()));
}

macro_rules! b {
    ($id:ident, $($args:expr),*) => {
        #[allow(non_snake_case)]
        fn $id(c: &mut Criterion) {
            b(c, stringify!($id), $($args),*);
        }
    }
}

b!(Conv2d_2a_3x3, 149, 149, 32, 3, 3, 32, 1, Valid);
b!(Conv2d_2b_3x3, 147, 147, 32, 3, 3, 64, 1, Same);

criterion_group!(benches, Conv2d_2a_3x3, Conv2d_2b_3x3,);
criterion_main!(benches);
