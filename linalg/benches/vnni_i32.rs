#![allow(dead_code)]
// Kernel-level benchmark: AVX-512 VNNI int8 GEMM (avx512vnni_mmm_i32_8x8, VPDPBUSD
// over the K=4-inner PackedI8K4 layout) vs the AVX2 int8 path (avx2_mmm_i32_8x8,
// vpmaddubsw-style widening). Both run the i8i8 packing (index 1) over the same
// M/K/N so the only difference is the matmul inner loop.
use criterion::*;
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec, MatMatMul};

fn run_kernel(be: &mut Bencher, mmm: &dyn MatMatMul, m: usize, k: usize, n: usize) {
    let a = Tensor::zero_dt(DatumType::I8, &[m, k]).unwrap();
    let b = Tensor::zero_dt(DatumType::I8, &[k, n]).unwrap();
    let (pack_a, pack_b) = &mmm.packings()[1];
    let pa = pack_a.prepare_one(&a, 1, 0).unwrap();
    let pb = pack_b.prepare_one(&b, 0, 1).unwrap();
    let mut scratch = unsafe { mmm.allocate_scratch_space() };
    be.iter_custom(|iters| {
        let mut dur = std::time::Duration::default();
        for _ in 0..iters {
            let t = std::time::Instant::now();
            unsafe {
                mmm.run_with_scratch_space(
                    m,
                    n,
                    scratch.as_mut(),
                    &[FusedSpec::AddMatMul {
                        a: AsInputValue::Borrowed(&*pa),
                        b: AsInputValue::Borrowed(&*pb),
                        packing: 1,
                    }],
                )
                .unwrap()
            };
            dur += t.elapsed();
        }
        dur
    });
}

fn benches(c: &mut Criterion) {
    if !std::is_x86_feature_detected!("avx512vnni") {
        eprintln!("avx512vnni not available, skipping");
        return;
    }
    use tract_linalg::x86_64_fma::mmm::*;
    for &(m, k, n) in
        &[(64usize, 256usize, 64usize), (256, 256, 256), (512, 512, 512), (1024, 1024, 64)]
    {
        let id = format!("{m}x{k}x{n}");
        let mut g = c.benchmark_group("vnni_i32/packed_packed");
        g.throughput(Throughput::Elements((m * k * n) as u64));
        g.bench_with_input(BenchmarkId::new("avx2", &id), &(m, k, n), |b, &(m, k, n)| {
            run_kernel(b, &*avx2_mmm_i32_8x8.mmm(), m, k, n)
        });
        g.bench_with_input(BenchmarkId::new("avx512vnni", &id), &(m, k, n), |b, &(m, k, n)| {
            run_kernel(b, &*avx512vnni_mmm_i32_8x8.mmm(), m, k, n)
        });
        g.finish();
    }
}

criterion_group!(g, benches);
criterion_main!(g);
