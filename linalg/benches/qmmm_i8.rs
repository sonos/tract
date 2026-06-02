// int8 -> i32 GEMM (qmmm_i32) microbench. A/B the SME SMOPA kernel vs the NEON
// fallback by running twice: default (SME) vs TRACT_SME_DISABLE=1 (arm64simd 8x8).
extern crate criterion;
use criterion::*;
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec};

use DatumType::I32;

fn qmmm(be: &mut criterion::Bencher, &(m, k, n): &(usize, usize, usize)) {
    unsafe {
        let mmm = tract_linalg::ops().mmm(I32, Some(m), Some(k), Some(n)).unwrap();
        // packing index 1 == i8i8 for both sme_qmmm_i32_32x32 and arm64simd_mmm_i32_8x8.
        let a = Tensor::zero::<i8>(&[m, k]).unwrap();
        let b = Tensor::zero::<i8>(&[k, n]).unwrap();
        let packing = &mmm.packings()[1];
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<i32>(&[m, n]).unwrap();
        be.iter(move || {
            mmm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul {
                        a: AsInputValue::Borrowed(&*pa),
                        b: AsInputValue::Borrowed(&*pb),
                        packing: 1,
                    },
                    FusedSpec::Store(mmm.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
                ],
            )
        });
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("qmmm_i8");
    g.sample_size(20);
    for &shape in &[
        (256usize, 256usize, 256usize),
        (512, 512, 512),
        (1024, 1024, 1024),
        (128, 768, 768),
        (384, 768, 768),
        (64, 2048, 2048),
    ] {
        let (m, k, n) = shape;
        g.throughput(Throughput::Elements((m * k * n) as u64));
        g.bench_function(format!("{m}x{k}x{n}"), |be| qmmm(be, &shape));
    }
    g.finish();
}

criterion::criterion_group!(benches, bench);
criterion::criterion_main!(benches);
