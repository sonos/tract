#[macro_use]
extern crate criterion;
extern crate ndarray;
#[macro_use]
extern crate tract_core;
extern crate tract_tensorflow;

use criterion::Criterion;

use tract_core::ops::nn::{Conv, DataFormat, FixedParamsConv, PaddingSpec};
use tract_core::ops::prelude::*;
use tract_tensorflow::ops::nn::conv2d::*;
use tract_tensorflow::ops::nn::local_patch::*;

use tract_core::ops::Op;

fn mk(sizes: &[usize]) -> Tensor {
    let data = ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
        .into_shape(sizes)
        .unwrap();
    Tensor::from(DtArray::from(data))
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Algo {
    Conv2d,
    Gen,
    Fixed,
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Padding {
    Same,
    Valid,
}

impl Algo {
    fn build(&self, padding: Padding) -> Box<Op> {
        match self {
            &Algo::Conv2d => Box::new(Conv2D::<f32>::new(if padding == Padding::Valid {
                LocalPatch::valid(1, 1)
            } else {
                LocalPatch::same(1, 1)
            })),
            &Algo::Gen => Box::new(Conv::new(
                DataFormat::NHWC,
                true,
                None,
                None,
                if padding == Padding::Valid {
                    PaddingSpec::Valid
                } else {
                    PaddingSpec::SameUpper
                },
                None,
                1,
            )),
            &Algo::Fixed => {
                let input: TVec<Tensor> = tvec![mk(&[1, 82, 1, 40]).into()];
                let kernel = mk(&[41, 1, 40, 128]);
                Box::new(
                    FixedParamsConv::new(
                        DataFormat::NHWC,
                        true,
                        tvec![1, 1],
                        tvec![1, 1],
                        if padding == Padding::Valid {
                            PaddingSpec::Valid
                        } else {
                            PaddingSpec::SameUpper
                        },
                        input[0].shape(),
                        kernel.to_array_view::<f32>().unwrap(),
                        None,
                        1,
                    ).unwrap(),
                )
            }
        }
    }
}

fn bench_conv(bencher: &mut Criterion) {
    let inputs = tvec![mk(&[1, 82, 1, 40]).into(), mk(&[41, 1, 40, 128]).into()];
    bencher.bench_function_over_inputs(
        "conv",
        move |b, &(algo, pad)| {
            let op = algo.build(*pad);
            b.iter(|| op.as_stateless().unwrap().eval(inputs.clone()).unwrap())
        },
        &[
            (Algo::Conv2d, Padding::Same),
            (Algo::Conv2d, Padding::Valid),
            (Algo::Gen, Padding::Same),
            (Algo::Gen, Padding::Valid),
            (Algo::Fixed, Padding::Same),
            (Algo::Fixed, Padding::Valid),
        ],
    );
}

criterion_group!(benches, bench_conv);
criterion_main!(benches);
