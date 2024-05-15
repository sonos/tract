extern crate criterion;
use criterion::*;
use tract_data::internal::*;
use tract_linalg::frame::mmm::FusedSpec;

use DatumType::F32;

fn mat_mul_smmm(be: &mut criterion::Bencher, &(m, k, n): &(usize, usize, usize)) {
    unsafe {
        let mm = tract_linalg::ops().mmm(F32, F32, F32, Some(m), Some(k), Some(n)).unwrap();
        let a = Tensor::zero::<f32>(tvec![m, k]).unwrap();
        let b = Tensor::zero::<f32>(tvec![k, n]).unwrap();
        let pa = mm.a_pack().pack_tensor(&a, 1, 0).unwrap();
        let pb = mm.b_pack().pack_tensor(&b, 0, 1).unwrap();

        let mut c = Tensor::zero::<f32>(tvec![m, n]).unwrap();
        be.iter(move || {
            mm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul { a: &*pa, b: &*pb },
                    FusedSpec::Store(mm.c_view(0, 1).wrap(&c.view_mut())),
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
