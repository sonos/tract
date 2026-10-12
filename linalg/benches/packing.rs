//! Activation packing with inline and four-worker executors, from small to bandwidth-sized inputs.
//! Compare revisions with `cargo bench -p tract-linalg --features multithread-mm --bench packing`
//! and Criterion's `--save-baseline` / `--baseline` options.

use std::hint::black_box;
use std::time::Duration;

use criterion::*;
use tract_data::internal::*;
use tract_linalg::multithread::{Executor, multithread_tract_scope};
use tract_linalg::pack::PackedFormat;

fn packing(c: &mut Criterion) {
    for (k, mn) in [(64, 600), (128, 1024), (288, 21609), (256, 65536)] {
        let mut group = c.benchmark_group(format!("packing/{k}x{mn}"));
        group.throughput(Throughput::Bytes((k * mn * 4) as u64));
        let format = PackedFormat::new(DatumType::F32, 8, 16);
        for (layout, shape, k_axis, mn_axis) in [("mn", [k, mn], 0, 1), ("k", [mn, k], 1, 0)] {
            let input = Tensor::from_shape(&shape, &vec![1f32; k * mn]).unwrap();
            for nth in [1, 4] {
                let executor = Executor::multithread(nth);
                group.bench_function(format!("{layout}/t{nth}"), |b| {
                    multithread_tract_scope(executor.clone(), || {
                        b.iter(|| {
                            black_box(
                                format.pack_tensor_view(&input.view(), k_axis, mn_axis).unwrap(),
                            )
                        });
                    });
                });
            }
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = packing
}
criterion_main!(benches);
