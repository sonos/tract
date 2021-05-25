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

criterion_group!(benches, inceptionv3_5b);
criterion_main!(benches);
