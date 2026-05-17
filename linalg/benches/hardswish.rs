use criterion::*;
use tract_data::prelude::*;

use tract_linalg::element_wise::ElementWiseKer;

fn hardswish_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("hardswish_f32");
    group.throughput(Throughput::Elements(1024));
    let mut input = unsafe { Tensor::uninitialized_aligned::<f32>(&[1024], 16).unwrap() };
    let input = unsafe { input.as_slice_mut_unchecked::<f32>() };
    group.bench_function("rust", |b| b.iter(|| rust_f32(input)));
    group.bench_function("linalg", |b| b.iter(|| linalg32(input)));
    #[cfg(target_arch = "aarch64")]
    group.bench_function("linalg-asm", |b| {
        b.iter(|| tract_linalg::arm64::arm64simd_hardswish_f32_8n::run(input, ()))
    });
}

#[inline(never)]
fn rust_f32(input: &mut [f32]) {
    const INV6: f32 = 1.0 / 6.0;
    for x in input {
        let relu6 = ((*x + 3.0).min(6.0)).max(0.0);
        *x = *x * relu6 * INV6;
    }
}

#[inline(never)]
fn linalg32(input: &mut [f32]) {
    (tract_linalg::ops().hardswish_f32)().run(input).unwrap();
}

criterion_group!(benches, hardswish_f32);
criterion_main!(benches);
