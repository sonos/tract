// Microbenchmark: AVX-512 (zmm, 16-wide) element-wise activation kernels vs
// their x86 predecessor.
//
//   sigmoid, tanh              : predecessor = FMA (256-bit, 8-wide) kernel
//   hardswish, leaky_relu,
//   silu, gelu                 : predecessor = generic scalar kernel
//                                (no FMA kernel exists on x86)
//
// All buffers are 64-byte aligned (AVX-512 alignment_bytes) and a multiple of
// 64 elements so every kernel's nr() divides the length. Criterion reports the
// distribution; the min of the samples is the relevant "min-of-N" number.

use criterion::*;
use tract_data::prelude::*;
use tract_linalg::element_wise::ElementWiseKer;

const N: usize = 1024;

fn aligned_input() -> Tensor {
    let mut t = unsafe { Tensor::uninitialized_aligned::<f32>(&[N], 64).unwrap() };
    let s = unsafe { t.as_slice_mut_unchecked::<f32>() };
    for (i, x) in s.iter_mut().enumerate() {
        *x = (i as f32 / 10.0).sin() * 5.0;
    }
    t
}

// Enable FTZ/DAZ (flush-to-zero, denormals-are-zero) for the whole process so
// repeated in-place application of a kernel to its own output cannot collapse
// into denormal arithmetic (extremely slow on x86) and distort the timing.
// Mirrors what the sigmoid/tanh kernels already do internally via MXCSR.
#[cfg(target_arch = "x86_64")]
fn enable_ftz_daz() {
    // Set MXCSR bit 15 (FTZ) and bit 6 (DAZ) directly; the safe intrinsic
    // wrappers are deprecated in favour of inline asm.
    unsafe {
        let mut mxcsr: u32 = 0;
        std::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut mxcsr);
        mxcsr |= (1 << 15) | (1 << 6);
        std::arch::asm!("ldmxcsr [{p}]", p = in(reg) &mxcsr);
    }
}

// In-place throughput, matching the convention of the existing element-wise
// benches (sigmoid.rs / silu.rs).
macro_rules! bench_pair {
    ($c:expr, $name:expr, $pred_label:expr, $pred:ty, $avx512:ty $(, $param:expr)?) => {{
        let mut group = $c.benchmark_group($name);
        group.throughput(Throughput::Elements(N as u64));
        let mut tp = aligned_input();
        let sp = unsafe { tp.as_slice_mut_unchecked::<f32>() };
        group.bench_function($pred_label, |b| {
            b.iter(|| <$pred>::run(sp, ($($param)?)))
        });
        if std::is_x86_feature_detected!("avx512f") {
            let mut ta = aligned_input();
            let sa = unsafe { ta.as_slice_mut_unchecked::<f32>() };
            group.bench_function("avx512", |b| {
                b.iter(|| <$avx512>::run(sa, ($($param)?)))
            });
        }
        group.finish();
    }};
}

fn benches(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    enable_ftz_daz();
    use tract_linalg::x86_64_fma::act::*;
    use tract_linalg::x86_64_fma::{
        avx512_sigmoid_f32, avx512_tanh_f32, fma_sigmoid_f32, fma_tanh_f32,
    };

    bench_pair!(c, "sigmoid_f32", "fma", fma_sigmoid_f32, avx512_sigmoid_f32);
    bench_pair!(c, "tanh_f32", "fma", fma_tanh_f32, avx512_tanh_f32);
    bench_pair!(
        c,
        "hardswish_f32",
        "generic",
        tract_linalg::generic::SHardSwish4,
        x86_64_avx512_hardswish_f32_64n
    );
    bench_pair!(
        c,
        "leaky_relu_f32",
        "generic",
        tract_linalg::generic::SLeakyRelu4,
        x86_64_avx512_leaky_relu_f32_64n,
        0.1f32
    );
    bench_pair!(
        c,
        "silu_f32",
        "generic",
        tract_linalg::generic::SSiLU4,
        x86_64_avx512_silu_f32_16n
    );
    bench_pair!(
        c,
        "gelu_f32",
        "generic",
        tract_linalg::generic::SGelu4,
        x86_64_avx512_gelu_f32_16n
    );
}

criterion_group!(g, benches);
criterion_main!(g);
