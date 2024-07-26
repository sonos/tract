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
b!(tract_including_packing);

pub fn tract_prepacked(
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
        let mmm = tract_linalg::ops().mmm(dt, Some(m), Some(k), Some(n)).unwrap();
        let (packer_a, packer_b) = mmm.packings()[0];
        let c_storage = mmm.c_view(0, 1);

        let mut scratch = mmm.allocate_scratch_space();
        let pa = packer_a.prepare_tensor(&a, 1, 0).unwrap();
        let pb = packer_b.prepare_tensor(&b, 0, 1).unwrap();

        crit.bench_function(&format!("tract_prepacked_{:?}", dt), |be| {
            be.iter(|| {
                mmm.run_with_scratch_space(
                    m,
                    n,
                    &mut *scratch,
                    &[
                        FusedSpec::AddMatMul { a: &*pa, b: &*pb, packing: 0 },
                        FusedSpec::Store(c_storage.wrap(&mut c.view_mut())),
                    ],
                )
                .unwrap()
            });
        });
    }
}

fn matmul(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut g = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    g.throughput(Throughput::Elements((m * k * n) as _));
    naive(&mut g, m, k, n);
    ctile_1x1(&mut g, m, k, n);
    tile_2x2(&mut g, m, k, n);
    ctile_2x2(&mut g, m, k, n);
    tile_4x4(&mut g, m, k, n);
    ctile_4x4(&mut g, m, k, n);
    cpacked_tile_4x4(&mut g, m, k, n);
    tile_8x8(&mut g, m, k, n);
    ctile_8x8(&mut g, m, k, n);
    cpacked_tile_8x8(&mut g, m, k, n);
    matrixmultiply(&mut g, m, k, n);
    cblas(&mut g, m, k, n);
    tract_including_packing(&mut g, m, k, n);
    tract_prepacked(&mut g, m, k, n, f32::datum_type());
    tract_prepacked(&mut g, m, k, n, f16::datum_type());
    g.finish();

    let mut g = c.benchmark_group(format!("mt/{}x{}x{}", m, k, n));
    g.throughput(Throughput::Elements((m * k * n) as _));
    let executor = tract_linalg::multithread::Executor::multithread_with_name(num_cpus::get(), "linalg-mt");
    tract_linalg::multithread::multithread_tract_scope(executor, || {
        tract_including_packing(&mut g, m, k, n);
        tract_prepacked(&mut g, m, k, n, f32::datum_type());
        tract_prepacked(&mut g, m, k, n, f16::datum_type());
    });
    g.finish();
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

fn whisper_base(c: &mut Criterion) {
    matmul(c, 512, 512, 1500);
}

fn big_matvec(c: &mut Criterion) {
    matmul(c, 4096, 4096, 1);
}

criterion_group!(benches, big, wavenet, asr_15M, inception, whisper_base, big_matvec);
criterion_main!(benches);
