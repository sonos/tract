use criterion::*;
use tract_data::internal::*;
use tract_linalg::frame::mmm::FusedSpec;

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
                    let mm =
                        tract_linalg::ops().mmm(F32, F32, F32, Some(m), Some(k), Some(1)).unwrap();
                    let a = Tensor::zero::<f32>(&[m, k]).unwrap();
                    let pa = mm.a_pack().pack_tensor(&a, 1, 0).unwrap();
                    let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
                    let pb = mm.b_pack().pack_tensor(&b, 0, 1).unwrap();
                    let mut c = Tensor::zero::<f32>(&[m]).unwrap();
                    be.iter(move || {
                        mm.run(
                            m,
                            1,
                            &[
                                FusedSpec::AddMatMul { a: &*pa, b: &*pb },
                                FusedSpec::Store(mm.c_view(0, 0).wrap(&c.view_mut())),
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
