use criterion::*;
use tract_data::prelude::*;
use tract_linalg::element_wise::ElementWiseKer;
use tract_linalg::generic::reduce::softmax_l2::{HSoftMaxL2, SSoftMaxL2};
use tract_linalg::reduce::{MapReduceKer, ReduceKer};

#[inline(never)]
fn loop1_f32_naive(slice: &mut [f32]) -> f32 {
    let mut max = f32::MIN;
    for x in &*slice {
        if *x > max {
            max = *x;
        }
    }
    max
}

#[inline(never)]
fn loop2_f32(slice: &mut [f32], max: f32) -> f32 {
    let mut sum = 0.;
    for x in slice.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    sum
}

#[inline(never)]
fn loop3_f32(slice: &mut [f32], sum: f32) {
    let recip = sum.recip();
    for x in slice {
        *x *= recip;
    }
}

#[inline(never)]
fn rust_f32(slice: &mut [f32]) {
    let max = loop1_f32_naive(slice);
    let sum = loop2_f32(slice, max);
    loop3_f32(slice, sum);
}

fn softmax_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_f32");
    // 1536 = 24*64 = 48*32: a multiple of both the FMA (32) and AVX-512 (64) tile
    // widths, 64-byte aligned so both kernels run entirely on their fast aligned
    // path (no prefix/suffix scalar fixup) for a fair before/after comparison.
    group.throughput(Throughput::Elements(1536));
    let mut input = unsafe { Tensor::uninitialized_aligned::<f32>(&[1536], 64).unwrap() };
    let mut plain = input.try_as_plain_mut().unwrap();
    let input = plain.as_slice_mut::<f32>().unwrap();
    // Deterministic finite values so every kernel sees identical, well-behaved
    // input (uninitialized memory could contain NaN/huge values that perturb the
    // fast-compact-exp int conversion and skew the comparison).
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i % 97) as f32) * 0.1 - 5.0;
    }
    group.bench_function("rust", |b| b.iter(|| rust_f32(input)));
    group.bench_function("loop1/naive", |b| b.iter(|| loop1_f32_naive(input)));
    group.bench_function("loop1/generic", |b| {
        b.iter(|| tract_linalg::generic::reduce::max::SMax4::red().run(input))
    });
    #[cfg(target_arch = "x86_64")]
    group.bench_function("loop1/iasm", |b| {
        b.iter(|| {
            tract_linalg::x86_64_fma::max::x86_64_fma_max_f32_32n::red().run(input).unwrap();
        })
    });
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        group.bench_function("loop1/avx512", |b| {
            b.iter(|| {
                tract_linalg::x86_64_fma::max::x86_64_avx512_max_f32_64n::red().run(input).unwrap();
            })
        });
    }
    #[cfg(target_arch = "aarch64")]
    group.bench_function("loop1/intr", |b| {
        b.iter(|| {
            tract_linalg::arm64::arm64simd_max_f32_16n::red().run(input).unwrap();
        })
    });
    group.bench_function("loop2/naive", |b| b.iter(|| loop2_f32(input, 1.0)));
    group.bench_function("loop2/generic", |b| {
        b.iter(|| SSoftMaxL2::red().run_with_params(input, 10.))
    });
    #[cfg(target_arch = "x86_64")]
    group.bench_function("loop2/iasm", |b| {
        b.iter(|| {
            tract_linalg::x86_64_fma::softmax::x86_64_fma_softmax2_fastcompact_f32_32n::red()
                .run_with_params(input, 10.)
                .unwrap()
        });
    });
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        group.bench_function("loop2/avx512", |b| {
            b.iter(|| {
                tract_linalg::x86_64_fma::softmax::x86_64_avx512_softmax2_fastcompact_f32_64n::red()
                    .run_with_params(input, 10.)
                    .unwrap()
            });
        });
    }
    #[cfg(target_arch = "aarch64")]
    group.bench_function("loop2/iasm", |b| {
        b.iter(|| {
            tract_linalg::arm64::arm64simd_softmax2_fastcompact_f32_16n::red()
                .run_with_params(input, 0.21)
                .unwrap()
        });
    });
    group.bench_function("loop3/naive", |b| b.iter(|| loop3_f32(input, 0.21)));
    group.bench_function("loop3/generic", |b| {
        b.iter(|| {
            tract_linalg::generic::by_scalar::SMulByScalar4::ew().run_with_params(input, 0.21)
        })
    });
    #[cfg(target_arch = "x86_64")]
    group.bench_function("loop3/iasm", |b| {
        b.iter(|| {
            tract_linalg::x86_64_fma::by_scalar::x86_64_avx_f32_mul_by_scalar_32n::ew()
                .run_with_params(input, 0.21)
                .unwrap()
        });
    });
    #[cfg(target_arch = "aarch64")]
    group.bench_function("loop3/iasm", |b| {
        b.iter(|| {
            tract_linalg::arm64::arm64simd_mul_by_scalar_f32_16n::ew()
                .run_with_params(input, 0.21)
                .unwrap()
        });
    });
}

fn softmax_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_f16");
    // 1536 = 64*24 (multiple of avx512 f16 nr=64 and generic h nr=8).
    const N: usize = 1536;
    group.throughput(Throughput::Elements(N as u64));
    let mut input = unsafe { Tensor::uninitialized_aligned::<f16>(&[N], 64).unwrap() };
    let mut plain = input.try_as_plain_mut().unwrap();
    let input = plain.as_slice_mut::<f16>().unwrap();
    for (i, x) in input.iter_mut().enumerate() {
        *x = f16::from_f32((i as f32 / 10.0).sin() * 5.0);
    }
    group.bench_function("loop2/generic", |b| {
        b.iter(|| HSoftMaxL2::red().run_with_params(input, f16::from_f32(10.0)))
    });
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") {
        group.bench_function("loop2/avx512", |b| {
            b.iter(|| {
                tract_linalg::x86_64_fma::softmax::x86_64_avx512_softmax2_fastcompact_f16_64n::red()
                    .run_with_params(input, f16::from_f32(10.0))
                    .unwrap()
            });
        });
    }
}

criterion_group!(benches, softmax_f32, softmax_f16);
criterion_main!(benches);
