use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::RngExt;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tract_hir::internal::*;
use tract_onnx::tract_core::dims;

fn bench_linear_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_linear_classifier");

    // Load model
    let model_path = "test_cases/linear_classifier/model.onnx";
    let onnx_path = PathBuf::from(&model_path);
    let model = tract_onnx::onnx().model_for_path(&onnx_path).unwrap();

    // Configure dimensions
    let n = model.sym("N");
    let model = model
        .with_input_fact(0, f32::fact(dims!(n, 12)).into())
        .unwrap()
        .with_output_fact(0, i64::fact(dims!(n)).into())
        .unwrap()
        .with_output_fact(1, f32::fact(dims!(n, 14)).into())
        .unwrap()
        .into_optimized()
        .unwrap();

    let input_fact = model.input_fact(0).unwrap().clone();
    let shape: TVec<usize> = input_fact
        .shape
        .as_concrete()
        .map(|s| s.iter().copied().collect())
        .unwrap_or_else(|| tvec![1, 12]);
    let num_features = shape[1];

    // Pre-generate random input tensors
    let mut rng = rand::rng();
    let input_tensors: Arc<Vec<Tensor>> = Arc::new(
        (0..1_000_000)
            .map(|_| {
                let sample: Vec<f32> =
                    (0..num_features).map(|_| rng.random_range(-30.0f32..30.0f32)).collect();
                Tensor::from_shape(&shape, &sample).unwrap()
            })
            .collect(),
    );

    let runnable = Arc::new(model.clone().into_runnable().unwrap());

    group.bench_function(BenchmarkId::new("load_opt_run_parallel", "per_call"), |b| {
        let runnable = Arc::clone(&runnable);
        let tensors = Arc::clone(&input_tensors);

        b.iter_custom(|iters| {
            let start = Instant::now();
            (0..iters as usize).into_par_iter().for_each(|i| {
                let input_val = tensors[i % 1_000_000].clone().into_tvalue();
                let _ = runnable.run(tvec!(input_val)).unwrap();
            });
            start.elapsed()
        });
    });

    group.finish();
}

criterion_group!(benches, bench_linear_classifier);
criterion_main!(benches);
