use criterion::*;
use tract_data::prelude::*;

use tract_linalg::element_wise::ElementWiseKer;

fn silu_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu_f32");
    group.throughput(Throughput::Elements(1024));
    // The per-arch entries call the kernels through ElementWiseKer::run, which skips
    // map_slice_with_alignment: the buffer must meet every kernel's alignment_bytes
    // itself (32 for the FMA ymm kernel, 64 for the AVX-512 zmm one) or they fault.
    let mut input = unsafe { Tensor::uninitialized_aligned::<f32>(&[1024], 64).unwrap() };
    let input = unsafe { input.as_slice_mut_unchecked::<f32>() };
    for (i, x) in input.iter_mut().enumerate() {
        *x = (i as f32 / 10.0).sin() * 5.0;
    }
    group.bench_function("rust_scalar", |b| b.iter(|| rust_scalar(input)));
    group.bench_function("linalg", |b| b.iter(|| linalg(input)));
    #[cfg(target_arch = "aarch64")]
    group.bench_function("linalg-asm-compose", |b| {
        b.iter(|| tract_linalg::arm64::arm64simd_silu_f32_4n::run(input, ()))
    });
    #[cfg(target_arch = "aarch64")]
    group.bench_function("linalg-asm-fused", |b| {
        b.iter(|| tract_linalg::arm64::arm64simd_silu_f32_4n_fused::run(input, ()))
    });
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") {
        group.bench_function("linalg-asm-fused", |b| {
            b.iter(|| tract_linalg::x86_64::fma_silu_f32::run(input, ()))
        });
    }
}

#[inline(never)]
fn rust_scalar(input: &mut [f32]) {
    for x in input {
        let sigmoid = 1.0 / (1.0 + (-*x).exp());
        *x *= sigmoid;
    }
}

#[inline(never)]
fn linalg(input: &mut [f32]) {
    tract_linalg::routines::Func::Silu.ew_f32().unwrap().run(input).unwrap();
}

#[cfg(target_arch = "aarch64")]
fn silu_f16(c: &mut Criterion) {
    for n in [1024usize, 65536, 1 << 20] {
        let mut group = c.benchmark_group(format!("silu_f16/{n}"));
        group.throughput(Throughput::Elements(n as u64));
        let mut input = unsafe { Tensor::uninitialized_aligned::<f16>(&[n], 16).unwrap() };
        let input = unsafe { input.as_slice_mut_unchecked::<f16>() };
        for (i, x) in input.iter_mut().enumerate() {
            *x = f16::from_f32((i as f32 / 10.0).sin() * 5.0);
        }
        group.bench_function("generic", |b| {
            b.iter(|| tract_linalg::generic::silu::generic_silu_f16_8n::run(input, ()))
        });
        group.bench_function("f32-roundtrip", |b| {
            b.iter(|| tract_linalg::arm64::arm64simd_silu_f16_4n::run(input, ()))
        });
        group.bench_function("lut", |b| {
            b.iter(|| tract_linalg::arm64::arm64simd_silu_f16_lut_8n::run(input, ()))
        });
        group.finish();
    }
}

#[cfg(target_arch = "aarch64")]
criterion_group!(benches, silu_f32, silu_f16);
#[cfg(not(target_arch = "aarch64"))]
criterion_group!(benches, silu_f32);
criterion_main!(benches);
