use activations::{definitions, reference, Program};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn crit(c: &mut Criterion, name: &str, r: impl Fn(f32) -> f32, prog: &Program) {
    let mut group = c.benchmark_group(name);
    for size in [1i32, 32, 256, 1024, 8192].iter() {
        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("Reference", size), size, |b, size| {
            b.iter_batched(
                || vec![1.0f32; *size as usize],
                |v| {
                    for x in v {
                        r(black_box(x));
                    }
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("VM", size), size, |b, size| {
            b.iter_batched(
                || vec![1.0f32; *size as usize],
                |v| {
                    for x in v {
                        prog.compute(black_box(x));
                    }
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("VMVec", size), size, |b, size| {
            b.iter_batched(
                || vec![1.0f32; *size as usize],
                |mut v| {
                    prog.compute_slice(black_box(&mut v));
                },
                BatchSize::LargeInput,
            )
        });
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    crit(c, "relu", reference::relu, &definitions::relu());
    crit(c, "hardswish", reference::hardswish, &definitions::hardswish());
    crit(c, "exp2f", reference::exp2f, &definitions::exp2f());
    crit(c, "sigmoid", reference::sigmoid, &definitions::sigmoid());
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
