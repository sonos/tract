use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tract_core::internal::*;
use tract_core::model::order::eval_order_for_nodes;
use tract_core::ops::array::TypedConcat;

fn eval_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_order");
    for n in [8, 2048] {
        let mut model = TypedModel::default();
        let inputs = (0..n)
            .map(|i| model.add_source(format!("input_{i}"), f32::fact([1])).unwrap())
            .collect::<Vec<_>>();
        let output = model.wire_node("concat", TypedConcat::new(0), &inputs).unwrap()[0];
        let input_nodes = inputs.iter().map(|input| input.node).collect::<Vec<_>>();
        group.bench_with_input(BenchmarkId::new("concat", n), &n, |b, _| {
            b.iter(|| {
                eval_order_for_nodes(
                    black_box(model.nodes()),
                    black_box(&input_nodes),
                    &[output.node],
                    &[],
                )
                .unwrap()
            });
        });
        let deps = (1..n).map(|i| (inputs[i].node, inputs[i - 1].node)).collect::<Vec<_>>();
        group.bench_with_input(BenchmarkId::new("extra_dependencies", n), &n, |b, _| {
            b.iter(|| {
                eval_order_for_nodes(
                    black_box(model.nodes()),
                    &[],
                    &[inputs[n - 1].node],
                    black_box(&deps),
                )
                .unwrap()
            });
        });
        let mut chain = TypedModel::default();
        let source = chain.add_source("input", f32::fact([1])).unwrap();
        let mut output = source;
        for i in 0..n {
            output = chain
                .wire_node(format!("add_{i}"), tract_core::ops::math::add(), &[output, output])
                .unwrap()[0];
        }
        group.bench_with_input(BenchmarkId::new("chain", n), &n, |b, _| {
            b.iter(|| {
                eval_order_for_nodes(black_box(chain.nodes()), &[source.node], &[output.node], &[])
                    .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, eval_order);
criterion_main!(benches);
