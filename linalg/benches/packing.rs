#![allow(non_snake_case)]

use criterion::measurement::WallTime;
use criterion::*;
use tract_data::internal::*;
use tract_linalg::frame::MatMatMulImpl;
use tract_linalg::mmm::FusedSpec;
use tract_linalg::mmm::MatMatMul;
use tract_linalg::mmm::MatMatMulKer;
use tract_linalg::mmm::ScratchSpaceFusedNonLinear;
use Throughput::Elements;

fn packa<K: MatMatMulKer<f32>>(crit: &mut BenchmarkGroup<WallTime>, m: usize, k: usize) {
    let a = Tensor::zero_dt(DatumType::F32, &[m, k]).unwrap();

    unsafe {
        let mmm = MatMatMulImpl::<K, f32>::new();
        let mut pa = Tensor::zero_aligned_dt(
            DatumType::F32,
            &[mmm.a_pack(k).len(m)],
            mmm.a_pack(k).alignment(),
        )
        .unwrap();

        crit.throughput(Elements((m * k) as _)).bench_function("packa", |be| {
            be.iter(|| mmm.a_pack(k).pack(&mut pa.view_mut(), &a.view(), 1, 0));
        });
    }
}

fn packb<K: MatMatMulKer<f32>>(crit: &mut BenchmarkGroup<WallTime>, k: usize, n: usize) {
    let b = Tensor::zero_dt(DatumType::F32, &[k, n]).unwrap();

    unsafe {
        let mmm = MatMatMulImpl::<K, f32>::new();
        let mut pb = Tensor::zero_aligned_dt(
            DatumType::F32,
            &[mmm.b_pack(k).len(n)],
            mmm.b_pack(k).alignment(),
        )
        .unwrap();

        crit.throughput(Elements((k * n) as _)).bench_function("packb", |be| {
            be.iter(|| mmm.b_pack(k).pack(&mut pb.view_mut(), &b.view(), 0, 1));
        });
    }
}

pub fn compute<
    K: MatMatMulKer<f32>,
    F: Fn(&mut ScratchSpaceFusedNonLinear<f32>, &[FusedSpec], usize, usize, usize),
>(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
    name: &str,
    f: F,
) {
    let mut c = Tensor::zero_dt(DatumType::F32, &[m, n]).unwrap();

    unsafe {
        let mmm = MatMatMulImpl::<K, f32>::new();
        let a_storage = mmm.a_packed(f32::datum_type().size_of(), k);
        let b_storage = mmm.b_packed(f32::datum_type().size_of(), k);
        let c_storage = mmm.c_view();

        let pa = Tensor::zero_aligned_dt(
            DatumType::F32,
            &[mmm.a_pack(k).len(m)],
            mmm.a_pack(k).alignment(),
        )
        .unwrap();
        let pb = Tensor::zero_aligned_dt(
            DatumType::F32,
            &[mmm.b_pack(k).len(n)],
            mmm.b_pack(k).alignment(),
        )
        .unwrap();
        let mut scratch = ScratchSpaceFusedNonLinear::<f32>::default();

        let ops = tvec!(
            FusedSpec::AddMatMul {
                k,
                a: a_storage.wrap(&pa.view()),
                b: b_storage.wrap(&pb.view()),
            },
            FusedSpec::Store(c_storage.wrap(&mut c.view_mut())),
        );

        crit.throughput(Elements((m * k * n) as _))
            .bench_function(name, |be| be.iter(|| f(&mut scratch, &ops, m, k, n)));
    }
}

fn prepacked_mr_nr<K: MatMatMulKer<f32>>(
    scratch: &mut ScratchSpaceFusedNonLinear<f32>,
    ops: &[FusedSpec],
    m: usize,
    _k: usize,
    n: usize,
) {
    unsafe {
        scratch.prepare::<K>(&ops);
        for ia in 0..m / K::mr() {
            for ib in 0..n / K::nr() {
                scratch.for_valid_tile::<K>(&ops, ia, ib);
                let err = K::kernel(&scratch.uspecs());
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
            }
        }
    }
}

fn prepacked_mc_nc_mr_nr<K: MatMatMulKer<f32>>(
    scratch: &mut ScratchSpaceFusedNonLinear<f32>,
    ops: &[FusedSpec],
    m: usize,
    _k: usize,
    n: usize,
) {
    unsafe {
        scratch.prepare::<K>(&ops);
        let mc = 128 - K::mr() % 128;
        let nc = 128 - K::nr() % 128;
        //    eprintln!("{}x{} {}x{} {}x{}", m, n, mc, nc, K::mr(), K::nr());
        for oa in 0..m.divceil(mc) {
            for ob in 0..n.divceil(nc) {
                for ia in 0..mc / K::mr() {
                    for ib in 0..nc / K::nr() {
                        let a = oa * mc / K::mr() + ia;
                        let b = ob * nc / K::nr() + ib;
                        if (a + 1) * K::mr() > m || (b + 1) * K::nr() > n {
                            continue;
                        }
                        scratch.for_valid_tile::<K>(&ops, a, b);
                        let err = K::kernel(&scratch.uspecs());
                        debug_assert_eq!(err, 0, "Kernel return error {}", err);
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
type K = tract_linalg::x86_64_fma::mmm::MatMatMulF32x16x6;

#[cfg(target_arch = "aarch64")]
type K = tract_linalg::arm64::MatMatMulF32x12x8;

fn matmul(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut c = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    packa::<K>(&mut c, m, k);
    packb::<K>(&mut c, k, n);
    compute::<K, _>(&mut c, m, k, n, "prepacked_mr_nr", prepacked_mr_nr::<K>);
    compute::<K, _>(&mut c, m, k, n, "prepacked_mc_nc_mr_nr", prepacked_mc_nc_mr_nr::<K>);
    c.finish();
}

fn big(c: &mut Criterion) {
    matmul(c, 512, 512, 512);
    #[cfg(target_arch = "x86_64")]
    matmul(c, 99, 891, 1048576);
    #[cfg(target_arch = "x86_64")]
    matmul(c, 128, 1024, 1048576);
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
