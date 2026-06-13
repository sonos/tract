//! End-to-end decode-loop benchmark in FOLDED KV-cache mode.
//!
//! Loads an NNEF LLM, applies `transformers_detect_all` (which folds attention
//! into stateful `DynKeyValueCache` ops) WITHOUT unfolding them, then drives a
//! real prefill + token-by-token decode loop through a persistent `SimpleState`.
//! Every decode step therefore goes through `resolve_symbols_with_states` over
//! all of the model's KV-cache op states — the path the `has_init_tensor_fact`
//! change optimizes.
//!
//! Reports allocations/token and ms/token (a counting global allocator measures
//! the decode window only, after warmup).
//!
//! Usage:
//!   cargo run --release --example llm_decode_bench -p tract-transformers -- <model.nnef.tgz> [prefill_len] [decode_tokens]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tract_nnef::internal::*;
use tract_nnef::tract_core::transform::get_transform;
use tract_transformers::WithTractTransformers;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

fn count_kv_caches(model: &TypedModel) -> usize {
    model
        .nodes()
        .iter()
        .filter(|n| n.op().name().contains("KeyValueCache") || n.op().name().contains("KvCache"))
        .count()
}

fn main() -> TractResult<()> {
    let mut args = std::env::args().skip(1);
    let model_path =
        args.next().expect("usage: llm_decode_bench <model.nnef.tgz> [prefill] [decode]");
    let prefill_len: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(16);
    let decode_tokens: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(64);

    eprintln!("Loading {model_path} ...");
    let t_load = Instant::now();
    let nnef = tract_nnef::nnef().with_tract_transformers();
    // Declutter BEFORE detection (matches the facade's load path), so the
    // Source -> Concat -> output KV-cache pattern is in canonical form.
    let mut model = nnef.model_for_path(&model_path)?.into_decluttered()?;

    // FOLD attention into stateful DynKeyValueCache ops (do NOT unfold).
    get_transform("transformers_detect_all")?
        .expect("transformers_detect_all not registered")
        .transform(&mut model)?;

    let n_kv = count_kv_caches(&model);
    let runnable = model.into_optimized()?.into_runnable()?;
    eprintln!("Loaded + optimized in {:?} ({n_kv} KV-cache ops)", t_load.elapsed());

    let mut state = runnable.spawn()?;

    // Token id input [1, S], i64. Values are irrelevant to decode timing.
    let prefill: TValue = Tensor::from_shape(&[1, prefill_len], &vec![1i64; prefill_len])?.into();
    let one_tok: TValue = Tensor::from_shape(&[1, 1], &[1i64])?.into();

    // Prefill once.
    let _ = state.run(tvec![prefill])?;

    // Warm up a few decode steps (prime allocator / caches / branch predictors).
    for _ in 0..4 {
        let o = state.run(tvec![one_tok.clone()])?;
        std::hint::black_box(&o);
    }

    // Measured decode window.
    let a0 = allocs();
    let t0 = Instant::now();
    let mut per_token_ns: Vec<u128> = Vec::with_capacity(decode_tokens);
    for _ in 0..decode_tokens {
        let ts = Instant::now();
        let o = state.run(tvec![one_tok.clone()])?;
        per_token_ns.push(ts.elapsed().as_nanos());
        std::hint::black_box(&o);
    }
    let elapsed = t0.elapsed();
    let a1 = allocs();

    let allocs_per_tok = (a1 - a0) as f64 / decode_tokens as f64;
    let ms_per_tok = elapsed.as_secs_f64() * 1e3 / decode_tokens as f64;
    let toks_per_s = decode_tokens as f64 / elapsed.as_secs_f64();
    per_token_ns.sort_unstable();
    let median_ms = per_token_ns[per_token_ns.len() / 2] as f64 / 1e6;

    println!(
        "--- E2E folded decode ({n_kv} KV-cache ops, prefill={prefill_len}, decode={decode_tokens}) ---"
    );
    println!("allocs/token : {allocs_per_tok:.1}");
    println!("ms/token     : {ms_per_tok:.3}  (median {median_ms:.3})");
    println!("tokens/sec   : {toks_per_s:.2}");
    Ok(())
}
