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

    use tract_linalg::x86_64_fma::mmm::*;

    // Representative large-K, square-ish M case.
    let (m, k) = (64usize, 256usize);

    // N = 5 :  current heuristic picks 64x3 (catch-all N<32). Zombie: 32x5.
    run(c, "N5_64x3_current", &*avx512_mmm_f32_64x3.mmm(), m, k, 5);
    run(c, "N5_32x5_zombie", &*avx512_mmm_f32_32x5.mmm(), m, k, 5);

    // N = 6 :  current heuristic picks 64x3. Zombie: 32x6.
    run(c, "N6_64x3_current", &*avx512_mmm_f32_64x3.mmm(), m, k, 6);
    run(c, "N6_32x6_zombie", &*avx512_mmm_f32_32x6.mmm(), m, k, 6);

    // N = 8 :  current heuristic picks 48x4 (N%4==0, N%3!=0). Zombie: 16x8.
    run(c, "N8_48x4_current", &*avx512_mmm_f32_48x4.mmm(), m, k, 8);
    run(c, "N8_16x8_zombie", &*avx512_mmm_f32_16x8.mmm(), m, k, 8);
}

criterion_group!(g, benches);
criterion_main!(g);
