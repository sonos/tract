//! Micro-probe for the per-run symbol-resolution overhead of stateful KV-cache
//! ops (`resolve_symbols_with_states`).
//!
//! Builds a compute-light model with `N` real `DynKeyValueCache` ops (mimicking
//! an N-layer transformer's K/V caches) and runs a short decode loop, reporting
//! HEAP ALLOCATIONS PER DECODE STEP and wall-time per step via a counting global
//! allocator. The KV-cache state persists across steps (one `SimpleState`), so
//! every step goes through `run_plan_with_eval` -> `resolve_symbols_with_states`.
//!
//! Run:  cargo run --release --example kv_resolve_probe -p tract-transformers
//!
//! Compare the `allocs/step` column before vs. after the `has_init_tensor_fact`
//! change: the old code called `init_tensor_fact()` (cloning a `String` name +
//! `TypedFact`) once per stateful op purely for an `is_some()` test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tract_nnef::internal::*;
use tract_transformers::ops::dyn_kv_cache::DynKeyValueCache;

// ---- counting global allocator -------------------------------------------------
struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(n, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

/// One source feeding `n_caches` `DynKeyValueCache` ops, all on a `head_dim`-wide
/// trailing axis, growing along axis 1 (sequence). Compute-light by design so the
/// per-run fixed overhead (incl. symbol resolution) is a visible fraction.
fn build(n_caches: usize, head_dim: usize) -> TractResult<Arc<TypedRunnableModel>> {
    let mut model = TypedModel::default();
    let s = model.sym("S");
    let p = model.sym("P");
    let input_shape: TVec<TDim> = tvec![1.to_dim(), s.into(), (head_dim as i64).to_dim()];
    let past_shape: TVec<TDim> = tvec![1.to_dim(), p.into(), (head_dim as i64).to_dim()];

    let input = model.add_source("input", f32::fact(&input_shape))?;
    let mut outs: TVec<OutletId> = tvec![];
    for i in 0..n_caches {
        // Realistic, allocation-forcing name (mirrors HF/Qwen NNEF cache names).
        let op = DynKeyValueCache {
            name: format!("model.layers.{i}.self_attn.kv_cache"),
            axis: 1,
            past_sequence_fact: f32::fact(&past_shape),
            input_sequence_fact: f32::fact(&input_shape),
        };
        let out = model.wire_node(format!("kv_{i}"), op, &[input])?;
        outs.push(out[0]);
    }
    model.select_output_outlets(&outs)?;
    model.into_runnable()
}

fn probe(n_caches: usize, head_dim: usize, steps: usize) -> TractResult<(f64, f64)> {
    let runnable = build(n_caches, head_dim)?;
    let mut state = runnable.spawn()?;

    // One decode token: [batch=1, seq=1, head_dim].
    let token: TValue = Tensor::zero::<f32>(&[1, 1, head_dim])?.into();

    // Warm up (first run resolves symbols / grows caches off the cold path).
    for _ in 0..3 {
        state.run(tvec![token.clone()])?;
    }

    let a0 = allocs();
    let t0 = Instant::now();
    for _ in 0..steps {
        let out = state.run(tvec![token.clone()])?;
        std::hint::black_box(&out);
    }
    let elapsed = t0.elapsed();
    let a1 = allocs();

    let allocs_per_step = (a1 - a0) as f64 / steps as f64;
    let ns_per_step = elapsed.as_nanos() as f64 / steps as f64;
    Ok((allocs_per_step, ns_per_step))
}

fn main() -> TractResult<()> {
    let head_dim = 8; // tiny -> concat is cheap -> fixed overhead is visible
    let steps = 64;

    println!("KV-cache resolve probe  (head_dim={head_dim}, steps/measurement={steps})");
    println!("{:>10} | {:>14} | {:>14}", "n_caches", "allocs/step", "ns/step");
    println!("{:-<10}-+-{:-<14}-+-{:-<14}", "", "", "");
    for &n in &[16usize, 32, 64, 128] {
        // a couple of repeats; report the best (least-noisy) timing, allocs are deterministic
        let mut best = (f64::MAX, f64::MAX);
        for _ in 0..5 {
            let r = probe(n, head_dim, steps)?;
            if r.1 < best.1 {
                best = r;
            }
        }
        println!("{:>10} | {:>14.1} | {:>14.0}", n, best.0, best.1);
    }
    Ok(())
}
