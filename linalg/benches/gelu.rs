#![allow(clippy::excessive_precision)]

use criterion::*;
use tract_data::prelude::*;

#[cfg(target_arch = "aarch64")]
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

fn gelu_f16(c: &mut Criterion) {
    for n in [1024usize, 65536, 1 << 20] {
        let mut group = c.benchmark_group(format!("gelu_f16/{n}"));
        group.throughput(Throughput::Elements(n as u64));
        let mut input = unsafe { Tensor::uninitialized_aligned::<f16>(&[n], 16).unwrap() };
        let input = unsafe { input.as_slice_mut_unchecked::<f16>() };
        for (i, x) in input.iter_mut().enumerate() {
            *x = f16::from_f32((i as f32 / 10.0).sin() * 5.0);
        }
        group.bench_function("generic", |b| {
            b.iter(|| tract_linalg::generic::HGelu8::run(input, ()))
        });
        group
            .bench_function("lut", |b| b.iter(|| tract_linalg::generic::HGeluLut8::run(input, ())));
        group.finish();
    }
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

criterion_group!(benches, gelu_f32, gelu_f16);
criterion_main!(benches);
