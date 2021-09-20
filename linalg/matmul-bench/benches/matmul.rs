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

fn naive(crit: &mut BenchmarkGroup<WallTime>, m: usize, k: usize, n: usize) {
    let a = vec![0f32; m * k];
    let b = vec![0f32; k * n];
    let mut c = vec![0f32; m * n];
    crit.bench_function("naive", |be| {
        be.iter(|| {
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += a[row * k + i] * b[i * n + col];
                    }
                    c[row * n + col] = sum;
                }
            }
        })
    });
}

fn tile_2x2(crit: &mut BenchmarkGroup<WallTime>, m: usize, k: usize, n: usize) {
    let a = vec![0f32; m * k];
    let b = vec![0f32; k * n];
    let mut c = vec![0f32; m * n];
    crit.bench_function("tile_2x2", |be| {
        be.iter(|| {
            for row in 0..m / 2 {
                for col in 0..n / 2 {
                    let mut sum00 = 0.0;
                    let mut sum01 = 0.0;
                    let mut sum10 = 0.0;
                    let mut sum11 = 0.0;
                    for i in 0..k {
                        let a0 = a[row * k + i];
                        let a1 = a[(row + 1) * k + i];
                        let b0 = b[i * n + col];
                        let b1 = b[i * n + col + 1];
                        sum00 += a0 * b0;
                        sum01 += a0 * b1;
                        sum10 += a1 * b0;
                        sum11 += a1 * b1;
                    }
                    c[row * n + col] = sum00;
                    c[row * n + col + 1] = sum01;
                    c[(row + 1) * n + col] = sum10;
                    c[(row + 1) * n + col + 1] = sum11;
                }
            }
        })
    });
}

fn matrixmultiply(crit: &mut BenchmarkGroup<WallTime>, m: usize, k: usize, n: usize) {
    let a = vec![0f32; m * k];
    let b = vec![0f32; k * n];
    let mut c = vec![0f32; m * n];
    crit.bench_function("matrixmultiply", |be| {
        be.iter(|| unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as _,
                1,
                b.as_ptr(),
                n as _,
                1,
                0.0,
                c.as_mut_ptr(),
                n as _,
                1,
            )
        });
    });
}

#[allow(unused_variables, unused_mut)]
fn cblas(crit: &mut BenchmarkGroup<WallTime>, m: usize, k: usize, n: usize) {
    let a = vec![0f32; m * k];
    let b = vec![0f32; k * n];
    let mut c = vec![0f32; m * n];
    #[cfg(feature = "blas")]
    crit.bench_function("blas", |be| {
        be.iter(|| unsafe {
            cblas::sgemm(
                cblas::Layout::RowMajor,
                cblas::Transpose::None,
                cblas::Transpose::None,
                m as _,
                n as _,
                k as _,
                1.0,
                &a,
                k as _,
                &b,
                n as _,
                0.0,
                &mut c,
                n as _,
            )
        })
    });
}

fn tract(crit: &mut BenchmarkGroup<WallTime>, m: usize, k: usize, n: usize) {
    use tract_linalg::frame::mmm::FusedSpec;
    let a = Tensor::zero_dt(DatumType::F32, &[m, k]).unwrap();
    let b = Tensor::zero_dt(DatumType::F32, &[k, n]).unwrap();
    let mut c = Tensor::zero_dt(DatumType::F32, &[n, m]).unwrap();

    unsafe {
        crit.bench_function("tract", |be| {
            let mmm = tract_linalg::ops()
                .mmm(DatumType::F32, DatumType::F32, DatumType::F32, Some(m), Some(k), Some(n))
                .unwrap();
            let a_storage = mmm.a_packed(f32::datum_type().size_of(), k);
            let b_storage = mmm.b_packed(f32::datum_type().size_of(), k);
            let c_storage = mmm.c_view_with_axis(1, 0);

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
            mmm.a_pack(k).pack(&mut pa.view_mut(), &a.view(), 1, 0);
            mmm.b_pack(k).pack(&mut pb.view_mut(), &b.view(), 0, 1);

            let mut scratch = mmm.allocate_scratch_space();

            be.iter(|| {
                mmm.run_with_scratch_space(
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
                .unwrap()
            });
        });
    }
}

fn matmul(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut c = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    c.throughput(Throughput::Elements((m * k * n) as _));
    naive(&mut c, m, k, n);
    tile_2x2(&mut c, m, k, n);
    matrixmultiply(&mut c, m, k, n);
    cblas(&mut c, m, k, n);
    tract(&mut c, m, k, n);
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
}

fn inception(c: &mut Criterion) {
    matmul(c, 64, 288, 21609);
}

criterion_group!(benches, big, wavenet, asr_15M, inception);
criterion_main!(benches);
