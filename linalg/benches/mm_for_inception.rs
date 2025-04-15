extern crate criterion;
use criterion::*;
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec};

use DatumType::F32;

fn mat_mul_smmm(be: &mut criterion::Bencher, &(m, k, n): &(usize, usize, usize)) {
    unsafe {
        let mmm = tract_linalg::ops().mmm(F32, Some(m), Some(k), Some(n)).unwrap();
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let b = Tensor::zero::<f32>(&[k, n]).unwrap();
        let packing = &mmm.packings()[0];
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();

        let mut c = Tensor::zero::<f32>(&[m, n]).unwrap();
        be.iter(move || {
            mmm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul {
                        a: AsInputValue::Borrowed(&*pa),
                        b: AsInputValue::Borrowed(&*pb),
                        packing: 0,
                    },
                    FusedSpec::Store(mmm.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
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
