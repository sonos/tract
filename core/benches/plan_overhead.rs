//! Plan-loop dispatch overhead micro-bench.
//!
//! Builds chains of N trivial ops (Add(unique-const)) and measures `plan.run()`
//! wall-time. Per-op dispatch overhead is `ns_per_run / n_nodes` for large N
//! (small N is dominated by run-entry cost).
//!
//! Use this to detect regressions in `do_exec_plan_with_eval`'s per-node loop.

use criterion::{Criterion, criterion_group, criterion_main};
use tract_core::internal::*;
use tract_core::ops::math::add;

const VEC_LEN: usize = 8;

fn build_chain(n: usize) -> Arc<TypedRunnableModel> {
    let mut model = TypedModel::default();
    let input_fact = f32::fact([VEC_LEN]);
    let mut prev = model.add_source("input", input_fact).unwrap();
    for i in 0..n {
        let v = (i as f32 + 1.0) * 1e-6;
        let c = model.add_const(format!("c{i}"), tensor1(&vec![v; VEC_LEN])).unwrap();
        prev = model.wire_node(format!("a{i}"), add(), &[prev, c]).unwrap()[0];
    }
    model.select_output_outlets(&[prev]).unwrap();
    model.into_optimized().unwrap().into_runnable().unwrap()
}

fn bench_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_overhead");
    let input: TValue = tensor1(&vec![1.0f32; VEC_LEN]).into();

    for &n in &[1, 10, 100, 1000] {
        let plan = build_chain(n);
        group.bench_function(format!("chain_n{n}"), |b| {
            b.iter(|| plan.run(tvec![input.clone()]).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_chain);
criterion_main!(benches);
