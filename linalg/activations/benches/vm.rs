use activations::{definitions, reference};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, BatchSize};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");
    for size in [1i32, 32, 256, 1024, 8192].iter() {
        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("Reference", size), size, |b, size| {
            b.iter_batched(
                || vec![1.0f32; *size as usize],
                |v| {
                    for x in v {
                        reference::relu(black_box(x));
                    }
                },
                BatchSize::LargeInput
                )
        });
        let d = definitions::relu();
        group.bench_with_input(BenchmarkId::new("VM", size), size, |b, size| {
            b.iter_batched(
                || vec![1.0f32; *size as usize],
                |v| {
                    for x in v {
                        d.compute(black_box(x));
                    }
                },
                BatchSize::LargeInput
                )
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
