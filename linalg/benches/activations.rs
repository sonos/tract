use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tract_linalg::frame::activations::{
    definitions, reference, Activation, ActivationKer, Op, Program,
};

const SIZES: &[i32] = &[32, 256, 1024, 8192];

fn vms() -> Vec<Box<dyn Activation<f32>>> {
    #[allow(unused_mut)]
    let mut vms = vec![tract_linalg::generic::activations::SActivations::act()];
    #[cfg(target_arch = "aarch64")]
    {
        vms.push(tract_linalg::arm64::arm64simd_act_f32_32n::act());
    }
    vms
}

fn e2e(c: &mut Criterion, name: &str, r: impl Fn(f32) -> f32, prog: &Program<f32>) {
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
        for vm in vms() {
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

fn e2e_benchmarks(c: &mut Criterion) {
    e2e(c, "relu", reference::relu, &definitions::relu());
    e2e(c, "hardswish", reference::hardswish, &definitions::hard_swish());
    e2e(c, "sigmoid", reference::sigmoid, &definitions::sigmoid());
}

fn unit(c: &mut Criterion, name: &str, prog: &Program<f32>) {
    let mut group = c.benchmark_group(format!("unit/{name}"));
    for size in SIZES {
        group.throughput(criterion::Throughput::Elements(*size as u64));
        for vm in vms() {
            group.bench_with_input(
                BenchmarkId::new(vm.name(), size),
                size,
                |b, size| {
                    b.iter_batched(
                        || vec![1.0f32; *size as usize],
                        |mut v| vm.run(prog, &mut v),
                        BatchSize::LargeInput,
                    )
                },
            );
        }
    }
}

fn unit_benchmarks(c: &mut Criterion) {
    unit(c, "empty", &Program { ops: vec![] });
    unit(c, "noop", &Program { ops: vec![Op::Noop] });
    unit(c, "noop2", &Program { ops: vec![Op::Noop, Op::Noop] });
    unit(c, "noop100", &Program { ops: vec![Op::Noop; 100] });
    unit(c, "add100", &Program { ops: vec![Op::AddConst(1.0); 100] });
    unit(c, "sub100", &Program { ops: vec![Op::SubConst(1.0); 100] });
    unit(c, "mul100", &Program { ops: vec![Op::MulConst(1.0); 100] });
    unit(c, "fma100", &Program { ops: vec![Op::FMA(1.0); 100] });
}

criterion_group!(benches, /* e2e_benchmarks, */ unit_benchmarks);
criterion_main!(benches);
