#![allow(dead_code)]
// Kernel-level benchmark: Intel AMX bf16 GEMM for f32 matmul
// (avx512amx_mmm_f32_16x16, TDPBF16PS over 16x16 f32 tile with K=32 bf16 inner)
// vs the AVX-512 f32 16x12 path (avx512_mmm_f32_16x12, FMA) vs the AVX2/FMA
// f32 16x6 path (fma_mmm_f32_16x6).
//
// The AMX path runs the f32f32_bf16 packing (index 1) which truncates f32 to
// bf16 at pack time (round-to-nearest-even, matching VCVTNEPS2BF16) so the f32
// accumulators carry the bf16 precision profile -- same trade-off as oneDNN
// "fast-math" f32 matmul on AMX. The two reference kernels run their default
// f32 packing (index 0).
//
// Skipped at runtime when has_amx_bf16() returns false (= CPUID lacks
// amx-bf16/tile or the arch_prctl XSAVE permission was denied), and at build
// time when the tract_amx_bf16 cfg was not emitted.
use criterion::*;
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec, MatMatMul};

fn run_kernel(be: &mut Bencher, mmm: &dyn MatMatMul, packing: usize, m: usize, k: usize, n: usize) {
    let a = Tensor::zero_dt(DatumType::F32, &[m, k]).unwrap();
    let b = Tensor::zero_dt(DatumType::F32, &[k, n]).unwrap();
    let (pack_a, pack_b) = &mmm.packings()[packing];
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
                        packing,
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
    #[cfg(tract_amx_bf16)]
    {
        use tract_linalg::x86_64_fma::amx_bf16::has_amx_bf16;
        use tract_linalg::x86_64_fma::mmm::*;
        if !has_amx_bf16() {
            eprintln!("AMX bf16 not available (CPUID + arch_prctl gate failed), skipping");
            return;
        }
        // Same shapes as amx_i32 so reviewers can directly compare bf16->f32 vs
        // i8->i32 throughput at matching M/K/N. K=32 (single tdpbf16ps step)
        // and K=64 (one i8 tile) are tested via 256 / 256x256 / 512x512x512.
        for &(m, k, n) in
            &[(64usize, 256usize, 64usize), (256, 256, 256), (512, 512, 512), (1024, 1024, 64)]
        {
            let id = format!("{m}x{k}x{n}");
            let mut g = c.benchmark_group("amx_f32/packed_packed");
            g.throughput(Throughput::Elements((m * k * n) as u64));
            // Reference: FMA f32 16x6 (the kernel mmm_f32 picks for these N).
            g.bench_with_input(BenchmarkId::new("fma_16x6", &id), &(m, k, n), |b, &(m, k, n)| {
                run_kernel(b, &*fma_mmm_f32_16x6.mmm(), 0, m, k, n)
            });
            if std::is_x86_feature_detected!("avx512f") {
                // Reference: AVX-512 f32 16x12.
                g.bench_with_input(
                    BenchmarkId::new("avx512_16x12", &id),
                    &(m, k, n),
                    |b, &(m, k, n)| run_kernel(b, &*avx512_mmm_f32_16x12.mmm(), 0, m, k, n),
                );
            }
            // AMX bf16 path (packing index 1 = f32f32_bf16: pack-time RNE
            // conversion of f32 -> bf16, then TDPBF16PS in the inner loop).
            g.bench_with_input(
                BenchmarkId::new("avx512amx_bf16_16x16", &id),
                &(m, k, n),
                |b, &(m, k, n)| run_kernel(b, &*avx512amx_mmm_f32_16x16.mmm(), 1, m, k, n),
            );
            g.finish();
        }
    }
    #[cfg(not(tract_amx_bf16))]
    {
        eprintln!("tract not built with AMX bf16 support (probe failed at build time)");
        let _ = c;
    }
}

criterion_group!(g, benches);
criterion_main!(g);
