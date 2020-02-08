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
    let kernel = Tensor::from(ndarray::Array4::<f32>::zeros((kh, kw, ci, co)));
    let conv = tract_core::hir::cnn::Conv::default()
        .nhwc()
        .hwio()
        .kernel_shape(kernel.shape()[0..2].into())
        .padding(padding)
        .strides(tvec!(stride, stride));
    let input_fact: TypedFact = TypedFact::dt_shape(DatumType::F32, image.shape()).unwrap();
    let kernel_fact: TypedFact = TypedFact::dt_shape(DatumType::F32, kernel.shape()).unwrap();
    let unary = conv.to_unary(&[&input_fact, &kernel_fact]).unwrap().unwrap();

    let mut m = TypedModel::default();
    let wire = m
        .add_source("", TypedFact::dt_shape(f32::datum_type(), [1, h, w, ci].as_ref()).unwrap())
        .unwrap();
    unsafe {
        unary.wire_as_im2col_pair(&mut m, "", wire, false).unwrap();
    }
    let im2col = m.node(1).op_as::<Im2Col<f32>>().unwrap();
    let args = tvec!(image.into());
    c.bench_function(name, move |b| {
        let ref op = im2col.as_stateless().unwrap();
        b.iter(|| op.eval(args.clone()).unwrap())
    });
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
