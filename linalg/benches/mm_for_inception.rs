extern crate criterion;
use criterion::*;
use tract_data::internal::*;
use tract_linalg::frame::mmm::FusedSpec;

use DatumType::F32;

fn mat_mul_smmm(be: &mut criterion::Bencher, &(m, k, n): &(usize, usize, usize)) {
    unsafe {
        let mm = tract_linalg::ops().mmm(F32, F32, F32, Some(m), Some(k), Some(n)).unwrap();
        let pa =
            Tensor::uninitialized_aligned::<f32>(&[mm.a_pack(k).len(m)], mm.a_pack(k).alignment())
                .unwrap();
        let pb =
            Tensor::uninitialized_aligned::<f32>(&[mm.b_pack(k).len(n)], mm.b_pack(k).alignment())
                .unwrap();
        let mut c = Tensor::zero::<f32>(&[m, n]).unwrap();
        be.iter(move || {
            mm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul {
                        a: mm.a_packed(F32.size_of(), k).wrap(&pa.view()),
                        b: mm.b_packed(F32.size_of(), k).wrap(&pb.view()),
                        k,
                    },
                    FusedSpec::Store(mm.c_view().wrap(&c.view_mut())),
                ],
            )
        });
    }
}

fn mat_mul_prepacked(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut group = c.benchmark_group("mat_mul_prepacked");
    group.bench_function("smmm", |be| mat_mul_smmm(be, &(m, k, n)));
}

fn s64x288x21609(c: &mut Criterion) {
    mat_mul_prepacked(c, 64, 288, 21609)
}

criterion::criterion_group!(benches, s64x288x21609);
criterion::criterion_main!(benches);
