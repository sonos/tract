//! Constant padding of small tensors and NCHW feature maps, inline and on four workers.

use std::time::Duration;

use criterion::*;
use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode};
use tract_linalg::multithread::{Executor, multithread_tract_scope};

fn pad(c: &mut Criterion) {
    for shape in [[1, 8, 4, 4], [1, 16, 160, 160], [1, 16, 320, 320], [1, 3, 640, 640]] {
        let mut group = c.benchmark_group(format!("pad/{shape:?}"));
        group.throughput(Throughput::Bytes((shape.iter().product::<usize>() * 4) as u64));
        let input =
            Tensor::from_shape(&shape, &vec![1f32; shape.iter().product()]).unwrap().into_tvalue();
        let op = Pad { pads: vec![(0, 0), (0, 0), (1, 1), (1, 1)], mode: PadMode::default() };
        for nth in [1, 4] {
            let executor = Executor::multithread(nth);
            group.bench_function(format!("t{nth}"), |b| {
                multithread_tract_scope(executor.clone(), || {
                    b.iter(|| op.eval_out_of_plan(tvec![input.clone()]).unwrap());
                });
            });
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = pad
}
criterion_main!(benches);
