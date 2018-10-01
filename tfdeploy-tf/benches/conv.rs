#[macro_use]
extern crate criterion;
extern crate ndarray;
#[macro_use]
extern crate tfdeploy;
extern crate tfdeploy_tf;

use criterion::Criterion;

use tfdeploy::ops::nn::{Conv, DataFormat, FixedParamsConv, PaddingSpec};
use tfdeploy::ops::Value;
use tfdeploy::*;
use tfdeploy_tf::ops::nn::conv2d::*;
use tfdeploy_tf::ops::nn::local_patch::*;

use tfdeploy::ops::Op;

fn mk(sizes: &[usize]) -> Value {
    let data = ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
        .into_shape(sizes)
        .unwrap();
    Value::from(Tensor::F32(data)).into_shared()
}

#[derive(Debug)]
enum Algo {
    Conv2d,
    Gen,
    Fixed,
}

impl Algo {
    fn build(&self, valid: bool) -> Box<Op> {
        match self {
            &Algo::Conv2d => Box::new(Conv2D::<f32>::new(if valid {
                LocalPatch::valid(1, 1)
            } else {
                LocalPatch::same(1, 1)
            })),
            &Algo::Gen => Box::new(Conv::new(
                DataFormat::NHWC,
                true,
                None,
                None,
                if valid {
                    PaddingSpec::Valid
                } else {
                    PaddingSpec::SameUpper
                },
                None,
            )),
            &Algo::Fixed => {
                let conv = Conv::new(
                    DataFormat::NHWC,
                    true,
                    None,
                    None,
                    if valid {
                        PaddingSpec::Valid
                    } else {
                        PaddingSpec::SameUpper
                    },
                    None,
                );
                let input: TVec<Value> = tvec![mk(&[1, 82, 1, 40]).into()];
                let kernel = mk(&[41, 1, 40, 128]);
                Box::new(
                    FixedParamsConv::new(
                        &conv,
                        input[0].shape(),
                        kernel.to_array_view::<f32>().unwrap(),
                        None,
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
        move |b, &(algo, valid)| {
            let op = algo.build(*valid);
            b.iter(|| op.eval(inputs.clone()).unwrap())
        },
        &[
        (Algo::Conv2d, false), (Algo::Conv2d, true),
        (Algo::Gen, false), (Algo::Gen, true),
        (Algo::Fixed, false), (Algo::Fixed, true),
        ],
    );
}

criterion_group!(benches, bench_conv);
criterion_main!(benches);
