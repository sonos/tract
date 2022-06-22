use std::time::Instant;

use tract_data::prelude::*;
use tract_linalg::frame::mmm::*;
use tract_linalg::mmm::OutputStoreKer;

fn ruin_cache() {
    return;
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
                vi[offset] = vi[offset] + a * b;
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

    let mut a =
        Tensor::zero_aligned::<T>(&[kernel.a_pack().len(k, m)], K::alignment_bytes_packed_a())
            .unwrap();

    let mut v = a.to_array_view_mut::<f32>().unwrap();
    v += 1.0;
    let mut b =
        Tensor::zero_aligned::<T>(&[kernel.b_pack().len(k, n)], K::alignment_bytes_packed_b())
            .unwrap();

    let mut v = b.to_array_view_mut::<f32>().unwrap();
    v += 1.0;
    let mut c = Tensor::zero::<T>(&[n, k]).unwrap();

    let ops = unsafe {
        [
            FusedSpec::AddMatMul {
                k,
                a: kernel.a_packed(4, k).wrap(&a.view()),
                b: kernel.b_packed(4, k).wrap(&b.view()).unwrap(),
            },
            // FusedSpec::AddUnicast(kernel.c_view(1, 0).wrap(&c.view_mut())),
            // FusedSpec::Store(kernel.c_view(1, 0).wrap(&c.view_mut())),
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
    let x = 17;
    let zp = bench_to_nanos::<T, K>(10, x, K::nr(), x);
    let y = 0.0; // bench_to_nanos::<T, K>(1000, x, K::nr(), x);
    let slope = (y - zp) / x as f64;
    (slope, zp)
}

fn as_match_line<T: Datum + Copy + num_traits::Zero + tract_linalg::LADatum, K: MatMatMulKer<T>>() {
    let coeffs = model::<T, K>();
    println!(
        "({:?}, {}, {}) => {} * k + {}  => ({} us)",
        K::name(),
        K::mr(),
        K::nr(),
        (coeffs.0 / 1000.).round(),
        (coeffs.1 / 1000.).round(),
        (coeffs.1 / 1000.0)
    );
}

fn main() {
    use tract_linalg::x86_64_fma::mmm::*;
    let core_id = core_affinity::get_core_ids().unwrap()[0];
    core_affinity::set_for_current(core_id);
    as_match_line::<f32, fma_mmm_f32_64x1>();
    as_match_line::<f32, avx512_mmm_f32_128x1>();
    as_match_line::<f32, avx512_mmm_f32_16x1>();
    // as_match_line::<f32, fma_mmm_f32_40x2>();
    // as_match_line::<f32, fma_mmm_f32_32x3>();
    // as_match_line::<f32, fma_mmm_f32_24x4>();
    // as_match_line::<f32, fma_mmm_f32_16x5>();
    // as_match_line::<f32, fma_mmm_f32_16x6>();
    // as_match_line::<f32, fma_mmm_f32_8x8>();
}
