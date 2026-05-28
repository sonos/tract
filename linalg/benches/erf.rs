// Microbenchmark: AVX-512 (zmm, 16-wide) erf kernel vs the generic scalar
// SErf4 (no FMA predecessor exists on x86). All buffers are 64-byte aligned
// (AVX-512 alignment_bytes) and a multiple of 64 elements so the kernel's
// nr() = 64 divides the length.

use criterion::*;
use tract_data::prelude::*;
use tract_linalg::element_wise::ElementWiseKer;

const N: usize = 1024;

fn aligned_input() -> Tensor {
    let mut t = unsafe { Tensor::uninitialized_aligned::<f32>(&[N], 64).unwrap() };
    let s = unsafe { t.as_slice_mut_unchecked::<f32>() };
    for (i, x) in s.iter_mut().enumerate() {
        *x = (i as f32 / 10.0).sin() * 5.0;
    }
    t
}

fn erf_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("erf_f32");
    g.throughput(Throughput::Elements(N as u64));
    let mut tp = aligned_input();
    let sp = unsafe { tp.as_slice_mut_unchecked::<f32>() };
    g.bench_function("generic", |b| b.iter(|| tract_linalg::generic::SErf4::run(sp, ())));
    if std::is_x86_feature_detected!("avx512f") {
        let mut ta = aligned_input();
        let sa = unsafe { ta.as_slice_mut_unchecked::<f32>() };
        g.bench_function("avx512", |b| {
            b.iter(|| tract_linalg::x86_64_fma::erf::x86_64_avx512_erf_f32_64n::run(sa, ()))
        });
    }
    g.finish();
}

criterion_group!(g, erf_f32);
criterion_main!(g);
