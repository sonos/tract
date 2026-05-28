// Microbench: AVX-512_FP16 native f16 element-wise activations vs the
// f32-roundtrip versions in `act_f16.rs` (which were the AVX-512 f16 path
// before native f16 ISA was available). Both run on 64-byte-aligned, 1024-
// element buffers — same workload as the existing activations_avx512_f16
// bench, just adding the native-fp16 column.

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

macro_rules! bench_triple {
    ($c:expr, $name:expr, $pred:ty, $roundtrip:ty, $native:ty $(, $param:expr)?) => {{
        let mut group = $c.benchmark_group($name);
        group.throughput(Throughput::Elements(N as u64));
        let mut tg = aligned_input();
        let sg = unsafe { tg.as_slice_mut_unchecked::<f16>() };
        group.bench_function("generic", |b| {
            b.iter(|| <$pred>::run(sg, ($($param)?)))
        });
        if std::is_x86_feature_detected!("avx512f") {
            let mut tr = aligned_input();
            let sr = unsafe { tr.as_slice_mut_unchecked::<f16>() };
            group.bench_function("avx512_f32roundtrip", |b| {
                b.iter(|| <$roundtrip>::run(sr, ($($param)?)))
            });
        }
        if std::is_x86_feature_detected!("avx512fp16") {
            let mut tn = aligned_input();
            let sn = unsafe { tn.as_slice_mut_unchecked::<f16>() };
            group.bench_function("avx512fp16_native", |b| {
                b.iter(|| <$native>::run(sn, ($($param)?)))
            });
        }
        group.finish();
    }};
}

fn benches(c: &mut Criterion) {
    bench_triple!(
        c,
        "hardswish_f16",
        tract_linalg::generic::hardswish::HHardSwish8,
        tract_linalg::x86_64_fma::act_f16::x86_64_avx512_hardswish_f16_64n,
        tract_linalg::x86_64_fma::act_f16_fp16::x86_64_avx512fp16_hardswish_f16_128n
    );
    bench_triple!(
        c,
        "leaky_relu_f16",
        tract_linalg::generic::leaky_relu::HLeakyRelu8,
        tract_linalg::x86_64_fma::act_f16::x86_64_avx512_leaky_relu_f16_64n,
        tract_linalg::x86_64_fma::act_f16_fp16::x86_64_avx512fp16_leaky_relu_f16_128n,
        f16::from_f32(0.1)
    );
}

criterion_group!(g, benches);
criterion_main!(g);
