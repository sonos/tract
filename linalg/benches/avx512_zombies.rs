#![allow(dead_code)]

use criterion::{Criterion, criterion_group, criterion_main};
use tract_linalg::mmm::MatMatMul;

#[path = "utils.rs"]
mod utils;
use utils::mat_mat_with_mm;

fn run(c: &mut Criterion, name: &str, mmm: &dyn MatMatMul, m: usize, k: usize, n: usize) {
    let mut group = c.benchmark_group(format!("avx512_zombie/{name}"));
    let id = format!("{m}x{k}x{n}");
    group.bench_with_input(
        criterion::BenchmarkId::new("hot", &id),
        &(tract_data::prelude::DatumType::F32, m, k, n, false),
        |b, p| mat_mat_with_mm(b, mmm, p),
    );
    group.bench_with_input(
        criterion::BenchmarkId::new("cold", &id),
        &(tract_data::prelude::DatumType::F32, m, k, n, true),
        |b, p| mat_mat_with_mm(b, mmm, p),
    );
}

fn benches(c: &mut Criterion) {
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("avx512f not available, skipping");
        return;
    }

    use tract_data::prelude::DatumType::F32;
    use tract_linalg::x86_64_fma::mmm::*;

    // Representative large-K, square-ish M case.
    let (m, k) = (64usize, 256usize);

    // N = 5 : zombie was 32x5 vs old 64x3.
    run(c, "N5_64x3_explicit", &*avx512_mmm_f32_64x3.mmm(), m, k, 5);
    run(c, "N5_32x5_explicit", &*avx512_mmm_f32_32x5.mmm(), m, k, 5);

    // N = 6 : zombie was 32x6 vs old 64x3.
    run(c, "N6_64x3_explicit", &*avx512_mmm_f32_64x3.mmm(), m, k, 6);
    run(c, "N6_32x6_explicit", &*avx512_mmm_f32_32x6.mmm(), m, k, 6);

    // N = 8 : zombie was 16x8 vs old 48x4.
    run(c, "N8_48x4_explicit", &*avx512_mmm_f32_48x4.mmm(), m, k, 8);
    run(c, "N8_16x8_explicit", &*avx512_mmm_f32_16x8.mmm(), m, k, 8);

    // What does the live dispatcher pick for these shapes? If the picker
    // is healthy these match the zombie numbers above, kernel name printed
    // to stderr at startup.
    for n in [5usize, 6, 8] {
        let mmm = tract_linalg::ops().mmm(F32, Some(m), Some(k), Some(n)).unwrap();
        eprintln!("dispatcher@m={m},k={k},n={n} picked {}", mmm.name());
        run(c, &format!("N{n}_dispatch"), &*mmm, m, k, n);
    }

    // Trace-only: a few shapes where M-padding overhead with the old
    // picker was high. We expect the M-aware picker to pick smaller-mr
    // kernels here.
    for (m, n) in [(20usize, 2), (33, 3), (50, 4), (17, 5), (1000, 64)] {
        let mmm = tract_linalg::ops().mmm(F32, Some(m), Some(k), Some(n)).unwrap();
        eprintln!("dispatcher@m={m},k={k},n={n} picked {}", mmm.name());
    }
}

criterion_group!(g, benches);
criterion_main!(g);
