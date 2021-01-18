extern crate criterion;
extern crate tract_core;
use criterion::*;

use nn::DataFormat::HWC;
use tract_core::internal::*;
use tract_core::ops::{cnn, nn};
use DatumType::F32;

fn im2col(c: &mut Criterion) {
    let mut group = c.benchmark_group("im2col");
    let pad = rctensor0(0.0f32);
    for dil in &[1, 2, 4, 8] {
        group.bench_function(&format!("dil_{}", dil), |b| {
            b.iter_with_setup(
                || {
                    let len = 8 + 2 * *dil;
                    let input = tvec!(
                        Tensor::zero_dt(f32::datum_type(), &[len, 16]).unwrap().into_arc_tensor(),
                        pad.clone()
                    );
                    let op = tract_core::ops::cnn::conv::Im2Col::new(
                        cnn::PatchSpec::for_full_shape(HWC, &[len, 16])
                            .unwrap()
                            .with_kernel_shape(tvec![3])
                            .with_dilations(tvec!(*dil))
                            .into_patch(),
                        HWC,
                        48,
                        8,
                        1,
                        16,
                        tract_linalg::ops().mmm(F32, F32, F32, 64, 48, 8).unwrap().b_pack(),
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

fn mmm(c: &mut Criterion) {
    c.bench_function("matmatmul", |b| {
        b.iter_with_setup(
            || {
                let mmm = tract_linalg::ops().mmm(F32, F32, F32, 64, 48, 8).unwrap();
                let packed_a = Tensor::zero_aligned_dt(
                    f32::datum_type(),
                    &[mmm.a_pack().len(64)],
                    mmm.a_pack().alignment(),
                )
                .unwrap();
                let input = tvec!(Tensor::zero_dt(f32::datum_type(), &[mmm.b_pack().len(8)])
                    .unwrap()
                    .into_arc_tensor());
                let op = tract_core::ops::matmul::lir_unary::LirMatMulUnary {
                    b_storage: unsafe { mmm.b_packed() },
                    c_fact: TypedFact::dt_shape(f32::datum_type(), &[8, 64]),
                    packed_as: tract_ndarray::arr0(packed_a.into_arc_tensor()).into_dyn(),
                    fused_ops: None,
                    mmm,
                    k: 48,
                    m: 64,
                    c_m_axis: 1,
                    c_n_axis: 0,
                    c_final_shape: (&[0, 64]).into(),
                };
                (input, op)
            },
            |(input, op)| {
                op.eval(input).unwrap();
            },
        )
    });
}

criterion_group!(benches, im2col, mmm);
criterion_main!(benches);
