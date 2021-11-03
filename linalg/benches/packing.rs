#![allow(non_snake_case)]

use criterion::measurement::WallTime;
use criterion::*;
use tract_data::internal::*;

#[derive(Copy, Clone)]
pub enum Bit {
    PackA,
    PackB,
    Compute,
}

pub fn run(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
    name: &str,
    bit: Bit,
) {
    use tract_linalg::frame::mmm::FusedSpec;
    let a = Tensor::zero_dt(DatumType::F32, &[m, k]).unwrap();
    let b = Tensor::zero_dt(DatumType::F32, &[k, n]).unwrap();
    let mut c = Tensor::zero_dt(DatumType::F32, &[m, n]).unwrap();

    unsafe {
        let mmm = tract_linalg::ops()
            .mmm(DatumType::F32, DatumType::F32, DatumType::F32, Some(m), Some(k), Some(n))
            .unwrap();
        let a_storage = mmm.a_packed(f32::datum_type().size_of(), k);
        let b_storage = mmm.b_packed(f32::datum_type().size_of(), k);
        let c_storage = mmm.c_view();

        let mut pa = Tensor::zero_aligned_dt(
            DatumType::F32,
            &[mmm.a_pack(k).len(m)],
            mmm.a_pack(k).alignment(),
        )
        .unwrap();
        let mut pb = Tensor::zero_aligned_dt(
            DatumType::F32,
            &[mmm.b_pack(k).len(n)],
            mmm.b_pack(k).alignment(),
        )
        .unwrap();
        let mut scratch = mmm.allocate_scratch_space();

        crit.bench_function(name, |be| {
            be.iter(|| match bit {
                Bit::PackA => mmm.a_pack(k).pack(&mut pa.view_mut(), &a.view(), 1, 0),
                Bit::PackB => mmm.b_pack(k).pack(&mut pb.view_mut(), &b.view(), 0, 1),
                Bit::Compute => mmm
                    .run_with_scratch_space(
                        m,
                        n,
                        &mut *scratch,
                        &[
                            FusedSpec::AddMatMul {
                                k,
                                a: a_storage.wrap(&pa.view()),
                                b: b_storage.wrap(&pb.view()),
                            },
                            FusedSpec::Store(c_storage.wrap(&mut c.view_mut())),
                        ],
                    )
                    .unwrap(),
            });
        });
    }
}

fn matmul(c: &mut Criterion, m: usize, k: usize, n: usize) {
    use Throughput::Elements;
    let mut c = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    run(&mut c.throughput(Elements((m * k) as _)), m, k, n, "packa", Bit::PackA);
    run(&mut c.throughput(Elements((k * n) as _)), m, k, n, "packb", Bit::PackB);
    run(&mut c.throughput(Elements((m * k * n) as _)), m, k, n, "compute", Bit::Compute);
    c.finish();
}

fn big(c: &mut Criterion) {
    matmul(c, 512, 512, 512);
    matmul(c, 99, 891, 1048576);
}

fn wavenet(c: &mut Criterion) {
    matmul(c, 32, 32, 8);
    matmul(c, 16, 60, 8);
}

fn asr_15M(c: &mut Criterion) {
    matmul(c, 768, 200, 24);
    matmul(c, 768, 2304, 24);
    matmul(c, 768, 2304, 8);
}

fn inception(c: &mut Criterion) {
    matmul(c, 64, 288, 21609);
}

criterion_group!(benches, big, wavenet, asr_15M, inception);
criterion_main!(benches);
