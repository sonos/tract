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
            &[mmm.a_pack().len(k, m)],
            mmm.a_pack().alignment(),
        )
        .unwrap();

        crit.throughput(Elements((m * k) as _)).bench_function("packa", |be| {
            be.iter(|| mmm.a_pack().pack(&mut pa.view_mut(), &a.view(), 1, 0));
        });
    }
}

fn packb<K: MatMatMulKer<f32>>(crit: &mut BenchmarkGroup<WallTime>, k: usize, n: usize) {
    let b = Tensor::zero_dt(DatumType::F32, &[k, n]).unwrap();

    unsafe {
        let mmm = MatMatMulImpl::<K, f32>::new();
        let mut pb = Tensor::zero_aligned_dt(
            DatumType::F32,
            &[mmm.b_pack().len(k, n)],
            mmm.b_pack().alignment(),
        )
        .unwrap();

        crit.throughput(Elements((k * n) as _)).bench_function("packb", |be| {
            be.iter(|| mmm.b_pack().pack(&mut pb.view_mut(), &b.view(), 0, 1));
        });
    }
}

fn packed_a<K: MatMatMulKer<f32>>(m: usize, k: usize) -> Tensor {
    let mmm = MatMatMulImpl::<K, f32>::new();
    Tensor::zero_aligned_dt(DatumType::F32, &[mmm.a_pack().len(k, m)], mmm.a_pack().alignment())
        .unwrap()
}

fn packed_b<K: MatMatMulKer<f32>>(k: usize, n: usize) -> Tensor {
    let mmm = MatMatMulImpl::<K, f32>::new();
    Tensor::zero_aligned_dt(DatumType::F32, &[mmm.b_pack().len(k, n)], mmm.b_pack().alignment())
        .unwrap()
}

#[inline(always)]
unsafe fn valid_tile(
    scratch: &mut ScratchSpaceFusedNonLinear<f32>,
    ops: &[FusedSpec],
    ia: usize,
    ib: usize,
) {
    scratch.for_valid_tile::<K>(ops, ia, ib);
    let err = K::kernel(&scratch.uspecs());
    debug_assert_eq!(err, 0, "Kernel return error {}", err);
}

unsafe fn packedpacked<K>(m: usize, k: usize, n: usize) -> (Tensor, Tensor, Tensor)
where
    K: MatMatMulKer<f32>,
{
    let c = Tensor::zero_dt(DatumType::F32, &[m, n]).unwrap();
    let pa = packed_a::<K>(m, k);
    let pb = packed_b::<K>(k, n);
    (pa, pb, c)
}

unsafe fn packedpacked_ops<'a, K: MatMatMulKer<f32>>(
    k: usize,
    pa: &'a Tensor,
    pb: &'a Tensor,
    c: &'a mut Tensor,
) -> TVec<FusedSpec<'a>> {
    let mmm = MatMatMulImpl::<K, f32>::new();
    tvec!(
        FusedSpec::AddMatMul {
            k,
            a: mmm.a_packed(4, k).wrap(&pa.view()),
            b: mmm.b_packed(4, k).wrap(&pb.view()).unwrap(),
        },
        FusedSpec::Store(mmm.c_view(0, 1).wrap(&mut c.view_mut())),
    )
}

unsafe fn packedpacking<K>(m: usize, k: usize, n: usize) -> (Tensor, Tensor, Tensor)
where
    K: MatMatMulKer<f32>,
{
    let c = Tensor::zero_dt(DatumType::F32, &[m, n]).unwrap();
    let pa = packed_a::<K>(m, k);
    let b = Tensor::zero_dt(DatumType::F32, &[k, n]).unwrap();
    (pa, b, c)
}

unsafe fn packedpacking_ops<'a, K: MatMatMulKer<f32>>(
    k: usize,
    pa: &'a Tensor,
    b: &'a Tensor,
    c: &'a mut Tensor,
) -> TVec<FusedSpec<'a>> {
    let mmm = MatMatMulImpl::<K, f32>::new();
    tvec!(
        FusedSpec::AddMatMul {
            k,
            a: mmm.a_packed(4, k).wrap(&pa.view()),
            b: mmm.b_late_packing().wrap(&b.view()).unwrap(),
        },
        FusedSpec::Store(mmm.c_view(0, 1).wrap(&mut c.view_mut())),
    )
}

fn packedpacked_mr_nr<K: MatMatMulKer<f32>>(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        let (pa, pb, mut c) = packedpacked::<K>(m, k, n);
        let ops = packedpacked_ops::<K>(k, &pa, &pb, &mut c);
        let mut scratch = ScratchSpaceFusedNonLinear::<f32>::default();
        crit.throughput(Elements((m * k * n) as _)).bench_function("packedpacked_mr_nr", |be| {
            be.iter(|| {
                scratch.prepare::<K>(&ops);
                for ia in 0..m / K::mr() {
                    for ib in 0..n / K::nr() {
                        valid_tile(&mut scratch, &ops, ia, ib);
                    }
                }
            })
        });
    }
}

fn packedpacked_nr_mr<K: MatMatMulKer<f32>>(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        let (pa, pb, mut c) = packedpacked::<K>(m, k, n);
        let ops = packedpacked_ops::<K>(k, &pa, &pb, &mut c);
        let mut scratch = ScratchSpaceFusedNonLinear::<f32>::default();
        crit.throughput(Elements((m * k * n) as _)).bench_function("packedpacked_nr_mr", |be| {
            be.iter(|| {
                scratch.prepare::<K>(&ops);
                for ib in 0..n / K::nr() {
                    for ia in 0..m / K::mr() {
                        valid_tile(&mut scratch, &ops, ia, ib);
                    }
                }
            })
        });
    }
}

fn packedpacking_mr_nr<K: MatMatMulKer<f32>>(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        let (pa, b, mut c) = packedpacking::<K>(m, k, n);
        let ops = packedpacking_ops::<K>(k, &pa, &b, &mut c);
        let mut scratch = ScratchSpaceFusedNonLinear::<f32>::default();
        crit.throughput(Elements((m * k * n) as _)).bench_function("packedpacking_mr_nr", |be| {
            be.iter(|| {
                scratch.prepare::<K>(&ops);
                for ia in 0..m / K::mr() {
                    for ib in 0..n / K::nr() {
                        valid_tile(&mut scratch, &ops, ia, ib);
                    }
                }
            })
        });
    }
}

fn packedpacking_nr_mr<K: MatMatMulKer<f32>>(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        let (pa, b, mut c) = packedpacking::<K>(m, k, n);
        let ops = packedpacking_ops::<K>(k, &pa, &b, &mut c);
        let mut scratch = ScratchSpaceFusedNonLinear::<f32>::default();
        crit.throughput(Elements((m * k * n) as _)).bench_function("packedpacking_nr_mr", |be| {
            be.iter(|| {
                scratch.prepare::<K>(&ops);
                for ib in 0..n / K::nr() {
                    for ia in 0..m / K::mr() {
                        valid_tile(&mut scratch, &ops, ia, ib);
                    }
                }
            })
        });
    }
}

fn packedpacked_mc_nc_mr_nr<K: MatMatMulKer<f32>>(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        let (pa, pb, mut c) = packedpacked::<K>(m, k, n);
        let ops = packedpacked_ops::<K>(k, &pa, &pb, &mut c);
        let mut scratch = ScratchSpaceFusedNonLinear::<f32>::default();
        crit.throughput(Elements((m * k * n) as _)).bench_function(
            "packedpacked_mc_nc_mr_nr",
            |be| {
                be.iter(|| {
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
                                    valid_tile(&mut scratch, &ops, ia, ib);
                                }
                            }
                        }
                    }
                })
            },
        );
    }
}

fn packedpacking_mc_nc_mr_nr<K: MatMatMulKer<f32>>(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        let (pa, b, mut c) = packedpacking::<K>(m, k, n);
        let ops = packedpacking_ops::<K>(k, &pa, &b, &mut c);
        let mut scratch = ScratchSpaceFusedNonLinear::<f32>::default();
        crit.throughput(Elements((m * k * n) as _)).bench_function(
            "packedpacking_mc_nc_mr_nr",
            |be| {
                be.iter(|| {
                    scratch.prepare::<K>(&ops);
                    let mc = 128 - K::mr() % 128;
                    let nc = 128 - K::nr() % 128;
                    //    eprintln!("{}x{} {}x{} {}x{}", m, n, mc, nc, K::mr(), K::nr());
                    for oa in 0..m.divceil(mc) {
                        for ob in 0..n.divceil(nc) {
                            for ib in 0..nc / K::nr() {
                                for ia in 0..mc / K::mr() {
                                    let a = oa * mc / K::mr() + ia;
                                    let b = ob * nc / K::nr() + ib;
                                    if (a + 1) * K::mr() > m || (b + 1) * K::nr() > n {
                                        continue;
                                    }
                                    valid_tile(&mut scratch, &ops, ia, ib);
                                }
                            }
                        }
                    }
                })
            },
        );
    }
}

#[cfg(target_arch = "x86_64")]
type K = tract_linalg::x86_64_fma::mmm::MatMatMulF32x16x6;

#[cfg(target_arch = "aarch64")]
type K = tract_linalg::arm64::MatMatMulF32x12x8;

#[cfg(target_arch = "arm")]
type K = tract_linalg::arm32::armv7neon::MatMatMulF32x8x6CortexA9;

fn matmul(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut c = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    packa::<K>(&mut c, m, k);
    packb::<K>(&mut c, k, n);
    packedpacked_mr_nr::<K>(&mut c, m, k, n);
    packedpacked_nr_mr::<K>(&mut c, m, k, n);
    packedpacked_mc_nc_mr_nr::<K>(&mut c, m, k, n);
    packedpacking_mr_nr::<K>(&mut c, m, k, n);
    packedpacking_nr_mr::<K>(&mut c, m, k, n);
    packedpacking_mc_nc_mr_nr::<K>(&mut c, m, k, n);
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
    matmul(c, 64, 48, 8);
    matmul(c, 16, 64, 8);
    matmul(c, 32, 64, 8);
}

fn asr_15M(c: &mut Criterion) {
    matmul(c, 768, 200, 18);
    matmul(c, 768, 2304, 18);
    matmul(c, 768, 2304, 6);
}

fn inception(c: &mut Criterion) {
    matmul(c, 64, 288, 21609);
}

criterion_group!(benches, big, wavenet, asr_15M, inception);
criterion_main!(benches);
