extern crate criterion;
extern crate tract_core;
use criterion::*;

use nn::DataFormat::HWC;
use tract_core::internal::*;
use tract_core::ops::matmul::lir_unary::*;
use tract_core::ops::{cnn, nn};
use DatumType::F32;

fn im2col(c: &mut Criterion) {
    let mut group = c.benchmark_group("im2col");
    let pad = rctensor0(0.0f32);
    for dil in &[1, 2, 4, 8] {
        group.bench_function(&format!("dil_{}", dil), |b| {
            b.iter_with_setup(
                || {
                    let pool_spec = tract_core::ops::cnn::PoolSpec {
                        data_format: HWC,
                        strides: None,
                        padding: cnn::PaddingSpec::Valid,
                        dilations: Some(tvec!(*dil)),
                        kernel_shape: tvec!(3),
                        output_channel_override: Some(64),
                    };
                    let len = 8 + 2 * *dil;
                    let input = tvec!(
                        Tensor::zero_dt(f32::datum_type(), &[len, 16]).unwrap().into_arc_tensor(),
                        pad.clone()
                    );
                    let mmm = tract_linalg::ops()
                        .mmm(F32, F32, F32, Some(64), Some(48), Some(8))
                        .unwrap();
                    let op = tract_core::ops::cnn::conv::Im2Col::new(
                        pool_spec,
                        1,
                        48,
                        &([len, 16].iter().collect()),
                        mmm,
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
                let mmm =
                    tract_linalg::ops().mmm(F32, F32, F32, Some(64), Some(48), Some(8)).unwrap();
                let packed_a = Tensor::zero_aligned_dt(
                    f32::datum_type(),
                    &[mmm.a_pack(48).len(64)],
                    mmm.a_pack(48).alignment(),
                )
                .unwrap();
                let input = tvec!(Tensor::zero_dt(f32::datum_type(), &[mmm.b_pack(48).len(8)])
                    .unwrap()
                    .into_arc_tensor());
                let geometry = MatMulGeometry::Concrete(ConcreteMatMulGeometry {
                    k: 48,
                    m: 64,
                    n: 8,
                    b_storage: unsafe { mmm.b_packed(F32.size_of(), 48) },
                });
                let op = LirMatMulUnary {
                    c_fact: TypedFact::dt_shape(f32::datum_type(), &[8, 64]),
                    micro_ops: tract_ndarray::arr0((
                        packed_a.into_arc_tensor(),
                        vec![ProtoFusedSpec::Store],
                    ))
                    .into_dyn(),
                    reshape_post: vec![],
                    c_m_axis: 1,
                    c_n_axis: 0,
                    c_final_shape: (&[0, 64]).into(),
                    geometry,
                    mmm,
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
