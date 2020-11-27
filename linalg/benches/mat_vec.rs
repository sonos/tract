use criterion::*;
use tract_data::internal::*;

use DatumType::F32;

fn mat_vec_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat_vec_mul");
    unsafe {
        for (m, k) in [(64usize, 64usize)].iter() {
            group.throughput(Throughput::Elements((m * k) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}", m, k)),
                &(m, k),
                |be, (&m, &k)| {
                    let mm = tract_linalg::ops().mmm(F32, F32, F32, m, k, 1).unwrap();
                    let pa = Tensor::uninitialized_aligned::<f32>(
                        &[mm.a_pack().len(m)],
                        mm.a_pack().alignment(),
                    )
                    .unwrap();
                    let b = tensor1(&vec![0.0; k]);
                    let mut c = Tensor::zero::<f32>(&[m]).unwrap();
                    mm.b_vec_from_data();
                    be.iter(move || {
                        mm.run(
                            &mm.a_packed().wrap(&pa.view()),
                            &mm.b_vec_from_data().wrap(&b.view()),
                            &mut mm.c_view().wrap(&c.view_mut()),
                            &[],
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, mat_vec_mul);
criterion_main!(benches);
