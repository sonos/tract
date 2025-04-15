use criterion::*;
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec};

use DatumType::F32;

fn mat_vec_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat_vec_mul");
    unsafe {
        {
            let (m, k) = &(768usize, 256usize);
            group.throughput(Throughput::Elements((m * k) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{m}x{k}")),
                &(m, k),
                |be, (&m, &k)| {
                    let mmm = tract_linalg::ops().mmm(F32, Some(m), Some(k), Some(1)).unwrap();
                    let packing = &mmm.packings()[0];
                    let a = Tensor::zero::<f32>(&[m, k]).unwrap();
                    let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
                    let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
                    let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
                    let mut c = Tensor::zero::<f32>(&[m]).unwrap();
                    be.iter(move || {
                        mmm.run(
                            m,
                            1,
                            &[
                                FusedSpec::AddMatMul {
                                    a: AsInputValue::Borrowed(&*pa),
                                    b: AsInputValue::Borrowed(&*pb),
                                    packing: 0,
                                },
                                FusedSpec::Store(mmm.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                            ],
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
