//! Matrix-vector products (n = 1) with f32 or f16 weights, per kernel, at three cache states:
//! `warm` (one weight panel, stays in L1/L2), `l3` (32 MB of panels in rotation) and `dram`
//! (512 MB). Pinned to one core, median of 9 batches. Plain output for comparing builds:
//! `cargo bench -p tract-linalg --bench f16_mat_vec [-- filter]`.
use std::hint::black_box;
use std::time::Instant;

use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec, MMMInputValue, MatMatMul};
use tract_linalg::pack::PackedFormat;

// DeepFilterNet 3 (768x256 dominates its bytes) and smaller layers off the 128-row tile.
const SHAPES: &[(usize, usize)] = &[(768, 256), (256, 256), (64, 64), (32, 64), (16, 32), (60, 16)];
const STATES: &[(&str, usize)] = &[("warm", 0), ("l3", 32 << 20), ("dram", 512 << 20)];

fn kernels() -> Vec<(&'static str, &'static dyn MatMatMul)> {
    #[allow(unused_mut)]
    let mut ks: Vec<(&'static str, &'static dyn MatMatMul)> = vec![];
    #[cfg(target_arch = "x86_64")]
    {
        use tract_linalg::x86_64::mmm::*;
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") {
            ks.push(("fma_32x1", &*fma_mmm_f32_32x1));
        }
        if is_x86_feature_detected!("avx512f") {
            ks.push(("avx512_128x1", &*avx512_mmm_f32_128x1));
            ks.push(("avx512_16x1", &*avx512_mmm_f32_16x1));
        }
    }
    ks
}

fn packing_for(mmm: &dyn MatMatMul, dt: DatumType) -> Option<usize> {
    mmm.packings().iter().position(|(a, b)| {
        a.downcast_ref::<PackedFormat>().is_some_and(|p| p.dt == dt)
            && b.downcast_ref::<PackedFormat>().is_some_and(|p| p.dt == f32::datum_type())
    })
}

fn time_one(
    mmm: &dyn MatMatMul,
    packing: usize,
    dt: DatumType,
    m: usize,
    k: usize,
    ws: usize,
) -> f64 {
    let (fa, fb) = &mmm.packings()[packing];
    let a = Tensor::from_shape(
        &[m, k],
        &(0..m * k).map(|i| (i % 13) as f32 * 0.01).collect::<Vec<_>>(),
    )
    .unwrap()
    .cast_to_dt(dt)
    .unwrap()
    .into_owned();
    let one = fa.prepare_one(&a, 1, 0).unwrap();
    let bytes = (m * k * dt.size_of()).max(1);
    let copies = (ws / bytes).max(1);
    let panels: Vec<Box<dyn MMMInputValue>> =
        (0..copies).map(|_| fa.prepare_one(&a, 1, 0).unwrap()).collect();
    drop(one);
    let b = Tensor::from_shape(&[k, 1], &vec![0.5f32; k]).unwrap();
    let pb = fb.prepare_one(&b, 0, 1).unwrap();
    let mut c = Tensor::zero::<f32>(&[m]).unwrap();
    let mut run = |i: usize| unsafe {
        mmm.run(
            m,
            1,
            &[
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&*panels[i % copies]),
                    b: AsInputValue::Borrowed(&*pb),
                    packing,
                },
                FusedSpec::Store(mmm.c_view(Some(0), Some(0)).wrap(&mut c.view_mut())),
            ],
        )
        .unwrap()
    };
    // Enough products for ~20 ms per batch, at least two laps over the panels.
    let t = Instant::now();
    for i in 0..200 {
        run(i);
    }
    let per = t.elapsed().as_secs_f64() / 200.0;
    let iters = ((0.02 / per) as usize).max(2 * copies).max(50);
    for i in 0..copies {
        run(i);
    }
    let mut batches: Vec<f64> = (0..9)
        .map(|_| {
            let t = Instant::now();
            for i in 0..iters {
                run(i);
            }
            t.elapsed().as_secs_f64() / iters as f64 * 1e9
        })
        .collect();
    batches.sort_by(|a, b| a.partial_cmp(b).unwrap());
    black_box(&c);
    batches[4]
}

fn main() {
    let filter = std::env::args().skip(1).find(|a| !a.starts_with('-')).unwrap_or_default();
    if let Some(core) = core_affinity::get_core_ids().and_then(|c| c.into_iter().nth(2)) {
        core_affinity::set_for_current(core);
    }
    for &(m, k) in SHAPES {
        for &(state, ws) in STATES {
            for (name, mmm) in kernels() {
                for dt in [f32::datum_type(), f16::datum_type()] {
                    let label = format!("{m}x{k} {state} {name} {dt:?}");
                    if !label.contains(&filter) {
                        continue;
                    }
                    let Some(p) = packing_for(mmm, dt) else { continue };
                    println!("{label}: {:.1} ns", time_one(mmm, p, dt, m, k, ws));
                }
            }
        }
    }
}
