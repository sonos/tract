//! Wall-clock scaling of the parallel MMM dispatcher across thread counts.
//!
//! Exercises the chunk-grid / cache-blocking split: each shape is run under a
//! rayon pool of 1, 2, 4 and 8 threads. t1 is the regression guard (the serial
//! path must stay neutral); the larger squares are where re-read traffic and
//! load-balance slack decide multi-thread scaling.
//!
//! Compare two revisions with criterion baselines:
//!   cargo bench -p tract-linalg --features multithread-mm --bench mm_threads -- --save-baseline main
//!   # switch revision
//!   cargo bench -p tract-linalg --features multithread-mm --bench mm_threads -- --baseline main
//!
//! `MM_THREADS` overrides the thread sweep (comma list), `MM_DT` picks `f32`
//! (default) or `i8`.

use criterion::*;
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec};
use tract_linalg::multithread::{Executor, multithread_tract_scope};

use DatumType::*;

/// (label, m, k, n). Element dims of the shapes in the PR's re-read table,
/// plus a tall-thin and a batched-ish wide case mirroring the harness cases.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("gliner_qkv_128", 128, 768, 768), // 16x96 panels @ 8x8
    ("ffn_up_256", 256, 768, 3072),    // 32x384 panels
    ("sq512", 512, 512, 512),          // 64x64
    ("sq1024", 1024, 1024, 1024),      // 128x128
    ("sq2048", 2048, 2048, 2048),      // interpolation point
    ("sq4096", 4096, 4096, 4096),      // 512x512 — headline re-read case
    ("tall_thin", 4096, 512, 128),     // skewed
    ("wide", 128, 512, 4096),          // skewed the other way
];

fn threads() -> Vec<usize> {
    match std::env::var("MM_THREADS") {
        Ok(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![1, 2, 4, 8],
    }
}

fn dt() -> DatumType {
    match std::env::var("MM_DT").as_deref() {
        Ok("i8") => I8,
        _ => F32,
    }
}

fn bench_shape(c: &mut Criterion, dt: DatumType, label: &str, m: usize, k: usize, n: usize) {
    let mut group = c.benchmark_group(format!("{label}/{m}x{k}x{n}"));
    group.throughput(Throughput::Elements((m * k * n) as u64));

    let mmm = tract_linalg::ops().policy_pick(dt, Some(m), Some(k), Some(n)).unwrap();
    let a = Tensor::zero_dt(dt, &[m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[k, n]).unwrap();
    let packing = &mmm.packings()[0];
    let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
    let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
    let mut cbuf = Tensor::zero_dt(dt, &[m, n]).unwrap();

    for &nth in &threads() {
        // A tract pool with <= 1 worker is treated by the dispatcher as "use the
        // global rayon pool" (mod.rs), so `multithread(1)` would silently run on
        // every core. The genuine serial path is `SingleThread`.
        let pool = if nth <= 1 { Executor::SingleThread } else { Executor::multithread(nth) };
        group.bench_function(BenchmarkId::from_parameter(format!("t{nth}")), |be| {
            let mut scratch = unsafe { mmm.allocate_scratch_space() };
            be.iter(|| {
                multithread_tract_scope(pool.clone(), || unsafe {
                    mmm.run_with_scratch_space(
                        m,
                        n,
                        scratch.as_mut(),
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(mmm.c_view(Some(0), Some(1)).wrap(&cbuf.view_mut())),
                        ],
                    )
                    .unwrap()
                })
            })
        });
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    let dt = dt();
    for &(label, m, k, n) in SHAPES {
        bench_shape(c, dt, label, m, k, n);
    }
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(800))
        .measurement_time(std::time::Duration::from_secs(4))
}

criterion::criterion_group! {
    name = benches_group;
    config = config();
    targets = benches
}
criterion::criterion_main!(benches_group);
