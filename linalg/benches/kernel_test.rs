use criterion::*;

mod utils;
use tract_data::prelude::DatumType;
use tract_linalg::mmm::MatMatMul;
use tract_linalg::mmm::MatMatMulKer;
use utils::*;

pub fn mat_mat_mm(
    be: &mut Bencher,
    &(mm, dt, m, k, n, cold): &(&dyn MatMatMul, DatumType, usize, usize, usize, bool),
) {
    mat_mat_with_mm(be, mm, &(dt, m, k, n, cold));
}

fn cold_and_hot(c: &mut Criterion, mm: &dyn MatMatMul, m: usize, k: usize, n: usize) {
    let mut group = c.benchmark_group(format!("{}", mm.kernel_name()));
    group.throughput(Throughput::Elements((m * k * n) as u64));
    let id = format!("{m}x{k}x{n}");
    group.bench_with_input(
        BenchmarkId::new("f32/cold", &id),
        &(mm, DatumType::F32, m, k, n, false),
        mat_mat_mm,
    );
    // group.bench_with_input(
    //     BenchmarkId::new("f32/hot", &id),
    //     &(mm, DatumType::F32, m, k, n, true),
    //     mat_mat_mm,
    // );
}

fn mm(be: &mut Criterion, mm: impl AsRef<dyn MatMatMul>, n: usize) {
    // for m in (0..1024).step_by(128).skip(1) {
    cold_and_hot(be, mm.as_ref(), 1024, 1000, n);
    // }
}

fn all(c: &mut Criterion) {
    use tract_linalg::x86_64_fma::mmm::*;
    macro_rules! benches_for_n {
        ($c:expr ; $n:expr ; $m:expr) => (
            paste::paste! {
                mm($c, [<avx512_mmm_f32_ $m x $n>]::mmm(), $n);
            }
        );
        ($c:expr ; $x:expr ; $m1:expr, $($y:expr),+) => (
            benches_for_n!($c ; $x ; $m1);
            benches_for_n!($c ; $x ; $($y),+);
        );
    }

    benches_for_n!(c; 1  ; 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240);
    benches_for_n!(c; 2  ; 16, 32, 48, 64, 80, 96, 112, 128, 144, 160);
    benches_for_n!(c; 3  ; 16, 32, 48, 64, 80, 96, 112);
    benches_for_n!(c; 4  ; 16, 32, 48, 64, 80, 96);
    benches_for_n!(c; 5  ; 16, 32, 48, 64, 80);
    benches_for_n!(c; 6  ; 16, 32, 48, 64);
    benches_for_n!(c; 7  ; 16, 32, 48);
    benches_for_n!(c; 8  ; 16, 32, 48);
    benches_for_n!(c; 9  ; 16, 32, 48);
    benches_for_n!(c; 10 ; 16, 32);
    benches_for_n!(c; 11 ; 16, 32);
    benches_for_n!(c; 12 ; 16, 32);
    benches_for_n!(c; 13 ; 16, 32);
    benches_for_n!(c; 14 ; 16, 32);
    benches_for_n!(c; 15 ; 16);
    benches_for_n!(c; 16 ; 16);
    benches_for_n!(c; 17 ; 16);
    benches_for_n!(c; 18 ; 16);
    benches_for_n!(c; 19 ; 16);
    benches_for_n!(c; 20 ; 16);
    benches_for_n!(c; 21 ; 16);
    benches_for_n!(c; 22 ; 16);
    benches_for_n!(c; 23 ; 16);
    benches_for_n!(c; 24 ; 16);
    benches_for_n!(c; 25 ; 16);
    benches_for_n!(c; 26 ; 16);
    benches_for_n!(c; 27 ; 16);
    benches_for_n!(c; 28 ; 16);
    benches_for_n!(c; 29 ; 16);
}

criterion_group!(benches, all);
criterion_main!(benches);
