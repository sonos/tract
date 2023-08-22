#![allow(dead_code)]
use std::time::Instant;

use tract_data::prelude::*;
use tract_linalg::frame::mmm::*;


fn ruin_cache() {
    // return;
    let _a = (0..1000000).collect::<Vec<i32>>();
}

pub fn reference<T, K>(mr: usize, k: usize, nr: usize) -> Vec<f32>
where
    T: Datum + Copy + num_traits::Zero + tract_linalg::LADatum,
    K: MatMatMulKer<T>,
{
    let mut vi = vec![0.0; k * nr];

    for m in 0..mr {
        for n in 0..nr {
            for _ in 0..k {
                let a: f32 = 1.0;
                let b = 1.0;
                let offset = { n + m * nr };
                vi[offset] += a * b;
            }
        }
    }
    vi
}

fn bench_to_nanos<
    T: Datum + Copy + num_traits::Zero + tract_linalg::LADatum,
    K: MatMatMulKer<T>,
>(
    loops: usize,
    m: usize,
    n: usize,
    k: usize,
) -> f64 {
    let kernel = K::mmm();

    let mut a = Tensor::zero_aligned::<T>(
        &[(k + K::end_padding_packed_a()) * m],
        K::alignment_bytes_packed_a(),
    )
    .unwrap();

    let mut v = a.to_array_view_mut::<f32>().unwrap();
    v += 1.0;
    let mut b = Tensor::zero_aligned::<T>(
        &[(k + K::end_padding_packed_b()) * n],
        K::alignment_bytes_packed_b(),
    )
    .unwrap();

    let mut v = b.to_array_view_mut::<f32>().unwrap();
    v += 1.0;
    let mut c = Tensor::zero::<T>(&[n, m]).unwrap();

    let ops = unsafe {
        [
            FusedSpec::AddMatMul {
                k,
                a: kernel.a_packed(4, k).wrap(&a.view()),
                b: kernel.b_packed(4, k).wrap(&b.view()),
            },
            // FusedSpec::AddUnicast(kernel.c_view(1, 0).wrap(&c.view_mut())),
            FusedSpec::Store(kernel.c_view(1, 0).wrap(&c.view_mut())),
        ]
    };

    let mut values = Vec::with_capacity(loops);

    for _ in 0..loops {
        ruin_cache();
        let start = Instant::now();
        unsafe { kernel.run(m, n, &ops).unwrap() };
        values.push(start.elapsed());
    }

    eprintln!("{:?} -> {:?}", values.first().unwrap(), values.last().unwrap());

    values.sort();
    values[loops / 2].as_nanos() as f64
}

fn model<T: Datum + Copy + num_traits::Zero + tract_linalg::LADatum, K: MatMatMulKer<T>>(
) -> (f64, f64) {
    let x = 1000;
    let zp = bench_to_nanos::<T, K>(1000, K::mr() * 4, K::nr() * 4, 0);
    let y = bench_to_nanos::<T, K>(1000, K::mr() * 4, K::nr() * 4, x);
    let slope = (y - zp) / x as f64;
    (slope, zp)
}

fn as_match_line<T: Datum + Copy + num_traits::Zero + tract_linalg::LADatum, K: MatMatMulKer<T>>() {
    let coeffs = model::<T, K>();
    println!("({:?}, {}, {}) => {} * k + {}", K::name(), K::mr(), K::nr(), (coeffs.0), (coeffs.1),);
}

fn main() {
    
    let core_id = core_affinity::get_core_ids().unwrap()[0];
    core_affinity::set_for_current(core_id);
    // as_match_line::<f32, fma_mmm_f32_64x1>();
    // as_match_line::<f32, avx512_mmm_f32_128x1>();
    // as_match_line::<f32, avx512_mmm_f32_16x1>();
    // as_match_line::<f32, fma_mmm_f32_40x2>();
    // as_match_line::<f32, fma_mmm_f32_32x3>();
    // as_match_line::<f32, fma_mmm_f32_24x4>();
    // as_match_line::<f32, fma_mmm_f32_16x5>();
    // as_match_line::<f32, fma_mmm_f32_16x6>();
    // as_match_line::<f32, fma_mmm_f32_8x8>();

    // mmv_perf_m();
    mmm_perf_batch_size();
}

// for mmv
fn mmv_perf_m() {
    use tract_linalg::x86_64_fma::mmm::*;
    let core_id = core_affinity::get_core_ids().unwrap()[0];
    core_affinity::set_for_current(core_id);
    fn bench<T: Datum + Copy + num_traits::Zero + tract_linalg::LADatum, K: MatMatMulKer<T>>(
        m: usize,
    ) {
        let val = bench_to_nanos::<T, K>(1000, m, 1, 100) / (m * 100) as f64;
        print!("{val}\t");
    }

    print!("N\t");
    print!("fma_mmm_f32_64x1\t");
    print!("avx512_mmm_f32_128x1\t");
    print!("avx512_mmm_f32_16x1\t");
    println!();
    for n in 1..=128 {
        eprintln!("{n}");
        print!("{n}\t");
        bench::<f32, fma_mmm_f32_64x1>(n);
        bench::<f32, avx512_mmm_f32_128x1>(n);
        bench::<f32, avx512_mmm_f32_16x1>(n);
        println!();
    }
}

// output a csv file with the perf of the kernels wrt batch size
fn mmm_perf_batch_size() {
    use tract_linalg::x86_64_fma::mmm::*;
    let core_id = core_affinity::get_core_ids().unwrap()[0];
    core_affinity::set_for_current(core_id);
    fn bench<T: Datum + Copy + num_traits::Zero + tract_linalg::LADatum, K: MatMatMulKer<T>>(
        n: usize,
    ) {
        let val =
            bench_to_nanos::<T, K>(1000, K::mr() * 4, n, 100) / (K::mr() * 4 * 100 * n) as f64;
        print!("{val}\t");
    }

    print!("N\t");
    print!("fma_mmm_f32_8x8\t");
    print!("fma_mmm_f32_16x6\t");
    print!("fma_mmm_f32_16x5\t");
    print!("fma_mmm_f32_24x4\t");
    print!("fma_mmm_f32_32x3\t");
    print!("fma_mmm_f32_40x2\t");
    print!("fma_mmm_f32_64x1\t");
    print!("avx512_mmm_f32_128x1\t");
    print!("avx512_mmm_f32_16x1\t");
    print!("avx512_mmm_f32_16x12\t");
    print!("avx512_mmm_f32_16x8\t");
    print!("avx512_mmm_f32_32x6\t");
    print!("avx512_mmm_f32_32x5\t");
    print!("avx512_mmm_f32_48x4\t");
    print!("avx512_mmm_f32_64x3\t");
    print!("avx512_mmm_f32_80x2\t");
    println!();
    for n in 1..=128 {
        eprintln!("{n}");
        print!("{n}\t");
        bench::<f32, fma_mmm_f32_8x8>(n);
        bench::<f32, fma_mmm_f32_16x6>(n);
        bench::<f32, fma_mmm_f32_16x5>(n);
        bench::<f32, fma_mmm_f32_24x4>(n);
        bench::<f32, fma_mmm_f32_32x3>(n);
        bench::<f32, fma_mmm_f32_40x2>(n);
        bench::<f32, fma_mmm_f32_64x1>(n);
        bench::<f32, avx512_mmm_f32_128x1>(n);
        bench::<f32, avx512_mmm_f32_16x1>(n);
        bench::<f32, avx512_mmm_f32_16x12>(n);
        bench::<f32, avx512_mmm_f32_16x8>(n);
        bench::<f32, avx512_mmm_f32_32x6>(n);
        bench::<f32, avx512_mmm_f32_32x5>(n);
        bench::<f32, avx512_mmm_f32_48x4>(n);
        bench::<f32, avx512_mmm_f32_64x3>(n);
        bench::<f32, avx512_mmm_f32_80x2>(n);
        println!();
    }
}
