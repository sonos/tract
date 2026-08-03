// f16 tanh on aarch64 cores without FEAT_FP16: generic scalar vs f32 roundtrip.
use criterion::*;
use tract_data::prelude::*;
use tract_linalg::element_wise::ElementWiseKer;

fn bench(c: &mut Criterion) {
    for n in [1024usize, 65536] {
        let mut t = unsafe { Tensor::uninitialized_aligned::<f16>(&[n], 16).unwrap() };
        let input = unsafe { t.as_slice_mut_unchecked::<f16>() };
        for (i, x) in input.iter_mut().enumerate() {
            *x = f16::from_f32((i as f32 / 10.0).sin() * 5.0);
        }
        let mut g = c.benchmark_group(format!("tanh_f16/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function("generic", |b| b.iter(|| tract_linalg::generic::HTanh8::run(input, ())));
        #[cfg(target_arch = "aarch64")]
        g.bench_function("f32-roundtrip", |b| {
            b.iter(|| tract_linalg::arm64::arm64simd_tanh_f16_4n::run(input, ()))
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
