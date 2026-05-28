// Microbench: fused RmsNorm vs the 4-call composition that tract-core currently
// uses (MeanOfSquares + Add + Rsqrt + Mul). The composition is reconstructed
// inline here in the same shape as `core::ops::nn::rms_norm::RmsNorm::eval`
// drives it. Both versions run on a 64-byte-aligned f32 row.

use criterion::*;
use tract_data::prelude::*;

fn aligned_row(n: usize) -> Tensor {
    let mut t = unsafe { Tensor::uninitialized_aligned::<f32>(&[n], 64).unwrap() };
    let s = unsafe { t.as_slice_mut_unchecked::<f32>() };
    for (i, x) in s.iter_mut().enumerate() {
        *x = (i as f32 / 10.0).sin() * 5.0;
    }
    t
}

#[inline(never)]
fn composed_rms_norm(buf: &mut [f32], eps: f32) {
    // Same shape as tract-core's RmsNorm::eval: separate passes for sum-of-squares,
    // mean, +eps, rsqrt, multiply — each writing/reading the row once.
    let mut sum_sq = 0.0_f32;
    for &x in buf.iter() {
        sum_sq += x * x;
    }
    let mean_sq = sum_sq / buf.len() as f32;
    let added = mean_sq + eps;
    let inv_std = added.sqrt().recip();
    for x in buf.iter_mut() {
        *x *= inv_std;
    }
}

fn rms_norm(c: &mut Criterion) {
    for &n in &[1024usize, 2048, 4096] {
        let id = format!("{n}");
        let mut g = c.benchmark_group(format!("rms_norm_f32/{id}"));
        g.throughput(Throughput::Elements(n as u64));
        let mut t = aligned_row(n);
        let s = unsafe { t.as_slice_mut_unchecked::<f32>() };
        g.bench_function("composed", |b| b.iter(|| composed_rms_norm(s, 1e-5)));
        g.bench_function("generic", |b| {
            b.iter(|| tract_linalg::generic::rms_norm::rms_norm_f32(s, 1e-5))
        });
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") {
            g.bench_function("avx512", |b| {
                b.iter(|| tract_linalg::x86_64_fma::rms_norm::rms_norm_f32(s, 1e-5))
            });
        }
        #[cfg(target_arch = "aarch64")]
        {
            g.bench_function("neon", |b| {
                b.iter(|| tract_linalg::arm64::arm64simd_rms_norm_f32(s, 1e-5))
            });
        }
        g.finish();
    }
}

criterion_group!(g, rms_norm);
criterion_main!(g);
