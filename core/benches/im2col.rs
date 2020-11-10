extern crate criterion;
extern crate tract_core;
use criterion::*;

use nn::DataFormat::HWC;
use tract_core::internal::*;
use tract_core::ops::{cnn, nn};

fn wavenet_dil(c: &mut Criterion) {
    let mut group = c.benchmark_group("wavenet");
    for dil in &[1, 2, 4, 8] {
        group.bench_function(&format!("dil_{}", dil), |b| {
            b.iter_with_setup(
                || {
                    let len = 8 + 2 * *dil;
                    let input = tvec!(Tensor::zero_dt(f32::datum_type(), &[len, 16])
                        .unwrap()
                        .into_arc_tensor());
                    let op = tract_core::ops::cnn::conv::Im2Col::new(
                        cnn::PatchSpec::for_full_shape(HWC, &[len, 16])
                            .unwrap()
                            .with_kernel_shape(tvec![3])
                            .with_dilations(tvec!(*dil))
                            .into_patch(),
                        HWC, // .shape(tvec![len, 16]).unwrap(),
                        64,
                        48,
                        8,
                        1,
                        16,
                        //        tract_linalg::ops().mmm(64, 48, 8, f32::datum_type(), f32::datum_type(), f32::datum_type()),
                        (tract_linalg::ops().mmm_f32)(64, 48, 8).b_pack(),
                        tensor0(0.0f32),
                    )
                    .unwrap();
                    (input, op)
                },
                |(input, op)| {
                    op.eval(input).unwrap();
                },
            )
        });
    }
}

criterion_group!(benches, wavenet_dil);
criterion_main!(benches);
