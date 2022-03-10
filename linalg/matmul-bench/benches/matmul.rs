#![allow(non_snake_case)]
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "blis")]
extern crate blis_src;
#[cfg(feature = "blis")]
extern crate cblas;

use criterion::measurement::WallTime;
use criterion::*;
use tract_data::internal::*;

macro_rules! b {
    ($id:ident) => {
        pub fn $id(crit: &mut BenchmarkGroup<WallTime>, m: usize, k: usize, n: usize) {
            let a = vec![0f32; m * k];
            let b = vec![0f32; k * n];
            let mut c = vec![0f32; m * n];
            crit.bench_function(stringify!($id), |be| {
                be.iter(|| matmul_bench::$id(m, k, n, &a, &b, &mut c))
            });
        }
    };
}

b!(naive);
b!(ctile_1x1);
b!(tile_2x2);
b!(ctile_2x2);
b!(tile_4x4);
b!(ctile_4x4);
b!(cpacked_tile_4x4);
b!(tile_8x8);
b!(ctile_8x8);
b!(cpacked_tile_8x8);
b!(matrixmultiply);
b!(cblas);
b!(tract);

pub fn tract_blaslike(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
    dt: DatumType,
) {
    use tract_linalg::frame::mmm::FusedSpec;
    let a = Tensor::zero_dt(dt, &[m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[k, n]).unwrap();
    let mut c = Tensor::zero_dt(dt, &[m, n]).unwrap();

    unsafe {
        let mmm = tract_linalg::ops().mmm(dt, dt, dt, Some(m), Some(k), Some(n)).unwrap();
        let a_storage = mmm.a_packed(f16::datum_type().size_of(), k);
        let b_storage = mmm.b_packed(f16::datum_type().size_of(), k);
        let c_storage = mmm.c_view(1, 0);

        let mut pa =
            Tensor::zero_aligned_dt(dt, &[mmm.a_pack().len(k, m)], mmm.a_pack().alignment())
                .unwrap();
        let mut pb =
            Tensor::zero_aligned_dt(dt, &[mmm.b_pack().len(k, n)], mmm.b_pack().alignment())
                .unwrap();
        let mut scratch = mmm.allocate_scratch_space();

        crit.bench_function(&format!("tract_blaslike_{:?}", dt), |be| {
            mmm.a_pack().pack(&mut pa.view_mut(), &a.view(), 1, 0);
            mmm.b_pack().pack(&mut pb.view_mut(), &b.view(), 0, 1);

            be.iter(|| {
                mmm.run_with_scratch_space(
                    m,
                    n,
                    &mut *scratch,
                    &[
                        FusedSpec::AddMatMul {
                            k,
                            a: a_storage.wrap(&pa.view()),
                            b: b_storage.wrap(&pb.view()).unwrap(),
                        },
                        FusedSpec::Store(c_storage.wrap(&mut c.view_mut())),
                    ],
                )
                .unwrap()
            });
        });
    }
}

fn matmul(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut c = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    c.throughput(Throughput::Elements((m * k * n) as _));
    naive(&mut c, m, k, n);
    ctile_1x1(&mut c, m, k, n);
    tile_2x2(&mut c, m, k, n);
    ctile_2x2(&mut c, m, k, n);
    tile_4x4(&mut c, m, k, n);
    ctile_4x4(&mut c, m, k, n);
    cpacked_tile_4x4(&mut c, m, k, n);
    tile_8x8(&mut c, m, k, n);
    ctile_8x8(&mut c, m, k, n);
    cpacked_tile_8x8(&mut c, m, k, n);
    matrixmultiply(&mut c, m, k, n);
    cblas(&mut c, m, k, n);
    tract(&mut c, m, k, n);
    tract_blaslike(&mut c, m, k, n, f32::datum_type());
    tract_blaslike(&mut c, m, k, n, f16::datum_type());
    c.finish();
}

fn big(c: &mut Criterion) {
    matmul(c, 512, 512, 512);
}

fn wavenet(c: &mut Criterion) {
    matmul(c, 32, 32, 8);
    matmul(c, 16, 60, 8);
}

fn asr_15M(c: &mut Criterion) {
    matmul(c, 768, 200, 24);
    matmul(c, 768, 2304, 24);
    matmul(c, 768, 2304, 8);
    matmul(c, 768, 384, 1);
}

fn inception(c: &mut Criterion) {
    matmul(c, 64, 288, 21609);
}

criterion_group!(benches, big, wavenet, asr_15M, inception);
criterion_main!(benches);
