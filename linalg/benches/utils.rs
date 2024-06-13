#![allow(dead_code)]
use criterion::*;
use tract_data::internal::*;
use tract_linalg::frame::mmm::{FusedSpec, MMMInputValue};
use tract_linalg::frame::MatMatMul;

use DatumType::*;

pub fn packed_packed(c: &mut Criterion, name: &str, m: usize, k: usize, n: usize) {
    let mut group = c.benchmark_group(format!("{name}/packed_packed"));
    group.throughput(Throughput::Elements((m * k * n) as u64));
    let id = format!("{m}x{k}x{n}");
    group.bench_with_input(BenchmarkId::new("f32/cold", &id), &(F32, m, k, n, true), mat_mat);
    group.bench_with_input(BenchmarkId::new("f32/hot", &id), &(F32, m, k, n, false), mat_mat);
    group.bench_with_input(BenchmarkId::new("i8/cold", &id), &(I8, m, k, n, true), mat_mat);
    group.bench_with_input(BenchmarkId::new("i8/hot", &id), &(I8, m, k, n, false), mat_mat);
}

pub fn packed_vec(c: &mut Criterion, name: &str, m: usize, k: usize, n: usize) {
    assert_eq!(n, 1);
    let mut group = c.benchmark_group(format!("{name}/packed_vec"));
    group.throughput(Throughput::Elements((m * k * n) as u64));
    let id = format!("{m}x{k}x{n}");
    group.bench_with_input(BenchmarkId::new("f32/cold", &id), &(F32, m, k, n, true), mat_mat);
    group.bench_with_input(BenchmarkId::new("f32/hot", &id), &(F32, m, k, n, false), mat_mat);
    group.bench_with_input(BenchmarkId::new("i8/cold", &id), &(I8, m, k, n, true), mat_mat);
    group.bench_with_input(BenchmarkId::new("i8/hot", &id), &(I8, m, k, n, false), mat_mat);
}

pub fn ruin_cache() {
    let _a = (0..1000000).collect::<Vec<i32>>();
}

#[allow(clippy::too_many_arguments)]
unsafe fn run(
    m: usize,
    _k: usize,
    n: usize,
    be: &mut Bencher,
    mmm: &dyn MatMatMul,
    a: &dyn MMMInputValue,
    b: &dyn MMMInputValue,
    cold: bool,
) {
    let mut scratch = mmm.allocate_scratch_space();
    be.iter_custom(move |iters| {
        let mut dur = std::time::Duration::default();
        for _ in 0..iters {
            if cold {
                ruin_cache();
            }
            let instant = std::time::Instant::now();
            mmm.run_with_scratch_space(
                m,
                n,
                scratch.as_mut(),
                &[FusedSpec::AddMatMul { a, b, packing: 0 }],
            )
            .unwrap();
            let time = instant.elapsed();
            dur += time;
        }
        dur
    });
}

fn mat_mat(be: &mut Bencher, params: &(DatumType, usize, usize, usize, bool)) {
    let (dt, m, k, n, _) = *params;
    let mm = tract_linalg::ops().mmm(dt, dt, dt, Some(m), Some(k), Some(n)).unwrap();
    mat_mat_with_mm(be, &*mm, params)
}

pub fn mat_mat_with_mm(
    be: &mut Bencher,
    mmm: &dyn MatMatMul,
    &(dt, m, k, n, cold): &(DatumType, usize, usize, usize, bool),
) {
    let a = Tensor::zero_dt(dt, &[m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[k, n]).unwrap();
    let packing = mmm.packings()[0];
    let pa = packing.0.prepare_tensor(&a, 1, 0).unwrap();
    let pb = packing.1.prepare_tensor(&b, 0, 1).unwrap();
    unsafe {
        run(m, k, n, be, mmm, &*pa, &*pb, cold);
    }
}
