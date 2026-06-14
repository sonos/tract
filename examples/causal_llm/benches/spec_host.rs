//! Host-side cost of the speculative-decoding glue: drafting, greedy
//! verification, and the per-position argmax over the vocabulary. These should
//! be negligible next to a forward pass (tens of milliseconds), confirming that
//! speculation's cost lives in the model, not the bookkeeping.

use causal_llm::speculative::{Drafter, NgramDrafter, argmax, greedy_verify};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    // argmax over a realistic vocabulary (Qwen3: 151936), run once per verified
    // position — the only host op with non-trivial cost.
    let vocab = 151_936u64;
    let logits: Vec<f32> =
        (0..vocab).map(|i| (i.wrapping_mul(2654435761) % 1000) as f32 * 0.01).collect();
    c.bench_function("argmax/vocab=151936", |b| b.iter(|| argmax(black_box(&logits))));

    // greedy verification of an 8-token draft.
    let drafts: Vec<u32> = (0..8).collect();
    let preds: Vec<u32> = (0..9).collect();
    c.bench_function("greedy_verify/k=8", |b| {
        b.iter(|| greedy_verify(black_box(&drafts), black_box(&preds)))
    });

    // n-gram drafting over a 1024-token context.
    let ctx: Vec<u32> = (0..1024).map(|i| (i % 50) as u32).collect();
    let mut drafter = NgramDrafter::default();
    c.bench_function("ngram_draft/ctx=1024,k=4", |b| {
        b.iter(|| drafter.draft(black_box(&ctx), 4).unwrap())
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
