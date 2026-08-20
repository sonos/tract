#[macro_use]
extern crate criterion;
extern crate tract_data;

use criterion::Criterion;
use tract_data::internal::*;

fn inceptionv3_5b(c: &mut Criterion) {
    c.bench_function("inceptionv3_5b", |b| {
        b.iter_with_setup(
            || unsafe {
                vec![
                    Tensor::uninitialized_dt(DatumType::F32, &[1, 35, 35, 64]).unwrap(),
                    Tensor::uninitialized_dt(DatumType::F32, &[1, 35, 35, 64]).unwrap(),
                    Tensor::uninitialized_dt(DatumType::F32, &[1, 35, 35, 96]).unwrap(),
                    Tensor::uninitialized_dt(DatumType::F32, &[1, 35, 35, 32]).unwrap(),
                ]
            },
            |input| Tensor::stack_tensors(3, &input),
        );
    });
}

// Recurrent-state stacking: the trailing axis is 1, so every block is a single
// f32 and the copy is `outer` of them. DTLN's model_1 and every FastEnhancer
// variant emit this shape once per frame.
fn one_datum_blocks(c: &mut Criterion) {
    for (name, shape) in
        [("dtln_state_1x2x128x1", [1, 2, 128, 1]), ("fastenhancer_state_1x256x1x1", [1, 256, 1, 1])]
    {
        c.bench_function(name, |b| {
            b.iter_with_setup(
                || unsafe {
                    vec![
                        Tensor::uninitialized_dt(DatumType::F32, &shape).unwrap(),
                        Tensor::uninitialized_dt(DatumType::F32, &shape).unwrap(),
                    ]
                },
                |input| Tensor::stack_tensors(3, &input),
            );
        });
    }
}

criterion_group!(benches, inceptionv3_5b, one_datum_blocks);
criterion_main!(benches);
