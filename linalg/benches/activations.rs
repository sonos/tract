use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tract_linalg::frame::activations::{definitions, reference, ActivationKer, Program};

const SIZES:&[i32] = &[32, 256, 1024, 8192];

fn crit(c: &mut Criterion, name: &str, r: impl Fn(f32) -> f32, prog: &Program<f32>) {
    let mut group = c.benchmark_group(name);
    for size in SIZES {
        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("Reference", size), size, |b, size| {
            b.iter_batched(
                || vec![1.0f32; *size as usize],
                |v| {
                    for x in v {
                        r(black_box(x));
                    }
                },
                BatchSize::LargeInput,
                )
        });
        #[allow(unused_mut)]
        let mut vms = vec!(tract_linalg::generic::activations::SActivations::act());
        #[cfg(target_arch="aarch64")]
        {
            vms.push(tract_linalg::arm64::arm64simd_act_f32_32n::act());
        }

        for vm in vms {
            group.bench_with_input(BenchmarkId::new(vm.name(), size), size, |b, size| {
                b.iter_batched(
                    || vec![1.0f32; *size as usize],
                    |mut v| vm.run(prog, &mut v),
                    BatchSize::LargeInput,
                    )
            });
        }
        if name == "sigmoid" {
            let sigmoid = (tract_linalg::ops().sigmoid_f32)();
            group.bench_with_input(BenchmarkId::new("handcrafted", size), size, |b, size| {
                b.iter_batched(
                    || vec![1.0f32; *size as usize],
                    |mut v| sigmoid.run(&mut v),
                    BatchSize::LargeInput,
                    )
            });
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    crit(c, "relu", reference::relu, &definitions::relu());
    crit(c, "hardswish", reference::hardswish, &definitions::hard_swish());
    crit(c, "sigmoid", reference::sigmoid, &definitions::sigmoid());

        /*
           crit(c, "exp2f", reference::exp2f, &definitions::exp2f());
           */
    }

    criterion_group!(benches, criterion_benchmark);
    criterion_main!(benches);
