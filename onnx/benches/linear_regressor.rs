use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::Rng;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tract_hir::internal::*;
use tract_onnx::tract_core::dims;

fn bench_linear_regressor(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_linear_regressor");

    // Load model
    let model_path = "test_cases/linear_regressor/model.onnx";
    let onnx_path = PathBuf::from(&model_path);
    let model = tract_onnx::onnx().model_for_path(&onnx_path).unwrap();

    // Configure dimensions
    let n = model.sym("N");
    let model = model
        .with_input_fact(0, f32::fact(dims!(n, 21)).into())
        .unwrap()
        .with_output_fact(0, f32::fact(dims!(n, 1)).into())
        .unwrap()
        .into_optimized()
        .unwrap();

    // Configure dimensions
    let input_fact = model.input_fact(0).unwrap().clone();
    let shape: TVec<usize> = input_fact
        .shape
        .as_concrete()
        .map(|s| s.iter().copied().collect())
        .unwrap_or_else(|| tvec![1, 21]);
    let num_features = shape[1];

    // Pre-generate random input tensors
    let mut rng = rand::thread_rng();
    let input_tensors: Arc<Vec<Tensor>> = Arc::new(
        (0..1_000_000)
            .map(|_| {
                let sample: Vec<f32> =
                    (0..num_features).map(|_| rng.gen_range(-30.0f32..30.0f32)).collect();
                Tensor::from_shape(&shape, &sample).unwrap()
            })
            .collect(),
    );

    let runnable = Arc::new(model.clone().into_runnable().unwrap());

    let iteration_counts = vec![1_000, 10_000, 100_000, 1_000_000];

    for &iterations in &iteration_counts {
        group.bench_function(BenchmarkId::new("load_opt_run_parallel", iterations), |b| {
            let runnable = Arc::clone(&runnable);
            let tensors = Arc::clone(&input_tensors);

            b.iter_custom(|_| {
                let start = Instant::now();

                (0..iterations).into_par_iter().for_each(|i| {
                    let runnable = Arc::clone(&runnable);
                    let input_val = tensors[i % 1_000_000].clone().into_tvalue();
                    let _ = runnable.run(tvec!(input_val)).unwrap();
                });

                start.elapsed()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_linear_regressor);
criterion_main!(benches);
