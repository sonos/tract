// Microbenchmark: AVX-512 f16 element-wise activations vs the generic scalar
// f16 kernels (no FMA f16 predecessor exists on x86 — the generic baseline
// already runs `(*v - max).to_f32()`-style per-element conversions). Buffers
// are 64-byte aligned (alignment that the AVX-512 path uses internally) and
// a multiple of 64 elements.

use criterion::*;
use tract_data::prelude::*;
use tract_linalg::element_wise::ElementWiseKer;

const N: usize = 1024;

fn aligned_input() -> Tensor {
    let mut t = unsafe { Tensor::uninitialized_aligned::<f16>(&[N], 64).unwrap() };
    let s = unsafe { t.as_slice_mut_unchecked::<f16>() };
    for (i, x) in s.iter_mut().enumerate() {
        *x = f16::from_f32((i as f32 / 10.0).sin() * 5.0);
    }
    t
}

macro_rules! bench_pair {
    ($c:expr, $name:expr, $pred:ty, $avx512:ty $(, $param:expr)?) => {{
        let mut group = $c.benchmark_group($name);
        group.throughput(Throughput::Elements(N as u64));
        let mut tp = aligned_input();
        let sp = unsafe { tp.as_slice_mut_unchecked::<f16>() };
        group.bench_function("generic", |b| {
            b.iter(|| <$pred>::run(sp, ($($param)?)))
        });
        if std::is_x86_feature_detected!("avx512f") {
            let mut ta = aligned_input();
            let sa = unsafe { ta.as_slice_mut_unchecked::<f16>() };
            group.bench_function("avx512", |b| {
                b.iter(|| <$avx512>::run(sa, ($($param)?)))
            });
        }
        group.finish();
    }};
}

fn benches(c: &mut Criterion) {
    bench_pair!(
        c,
        "sigmoid_f16",
        tract_linalg::generic::sigmoid::HSigmoid8,
        tract_linalg::x86_64_fma::act_f16::x86_64_avx512_sigmoid_f16_16n
    );
    bench_pair!(
        c,
        "tanh_f16",
        tract_linalg::generic::tanh::HTanh8,
        tract_linalg::x86_64_fma::act_f16::x86_64_avx512_tanh_f16_16n
    );
    bench_pair!(
        c,
        "hardswish_f16",
        tract_linalg::generic::hardswish::HHardSwish8,
        tract_linalg::x86_64_fma::act_f16::x86_64_avx512_hardswish_f16_64n
    );
    bench_pair!(
        c,
        "leaky_relu_f16",
        tract_linalg::generic::leaky_relu::HLeakyRelu8,
        tract_linalg::x86_64_fma::act_f16::x86_64_avx512_leaky_relu_f16_64n,
        f16::from_f32(0.1)
    );
    bench_pair!(
        c,
        "silu_f16",
        tract_linalg::generic::silu::HSiLU8,
        tract_linalg::x86_64_fma::act_f16::x86_64_avx512_silu_f16_16n
    );
    bench_pair!(
        c,
        "gelu_f16",
        tract_linalg::generic::gelu::HGelu8,
        tract_linalg::x86_64_fma::act_f16::x86_64_avx512_gelu_f16_16n
    );
}

criterion_group!(g, benches);
criterion_main!(g);
