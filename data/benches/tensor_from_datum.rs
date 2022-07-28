#[macro_use]
extern crate criterion;
extern crate tract_data;

use criterion::Criterion;
use tract_data::internal::*;

fn rank_4(c: &mut Criterion) {
    c.bench_function("rank_4", |b| {
        b.iter_with_setup(
            || {
                tract_ndarray::Array4::from_shape_simple_fn((256, 35, 35, 1), || 1.0f32)
                    .permuted_axes([3, 2, 1, 0])
            },
            Tensor::from,
        );
    });
}

criterion_group!(benches, rank_4);
criterion_main!(benches);
