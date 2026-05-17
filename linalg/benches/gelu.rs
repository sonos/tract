use criterion::*;
use tract_data::prelude::*;

use tract_linalg::element_wise::ElementWiseKer;

fn gelu_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu_f32");
    group.throughput(Throughput::Elements(1024));
    let mut input = unsafe { Tensor::uninitialized_aligned::<f32>(&[1024], 16).unwrap() };
    let input = unsafe { input.as_slice_mut_unchecked::<f32>() };
    for (i, x) in input.iter_mut().enumerate() {
        *x = (i as f32 / 10.0).sin() * 5.0;
    }
    group.bench_function("rust_scalar", |b| b.iter(|| rust_scalar(input)));
    group.bench_function("linalg", |b| b.iter(|| linalg(input)));
    #[cfg(target_arch = "aarch64")]
    group.bench_function("linalg-asm-compose", |b| {
        b.iter(|| tract_linalg::arm64::arm64simd_gelu_f32_4n::run(input, ()))
    });
    #[cfg(target_arch = "aarch64")]
    group.bench_function("linalg-asm-fused", |b| {
        b.iter(|| tract_linalg::arm64::arm64simd_gelu_f32_4n_fused::run(input, ()))
    });
}

#[inline(never)]
fn rust_scalar(input: &mut [f32]) {
    // Match tract's GeluApproximate scalar formula (pow=3).
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEF: f32 = 0.044715;
    for x in input {
        let v = *x;
        let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
        *x = 0.5 * v * (1.0 + inner.tanh());
    }
}

#[inline(never)]
fn linalg(input: &mut [f32]) {
    (tract_linalg::ops().gelu_f32)().run(input).unwrap();
}

criterion_group!(benches, gelu_f32);
criterion_main!(benches);
