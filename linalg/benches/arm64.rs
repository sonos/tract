use std::time::Instant;

use tract_data::prelude::*;
use tract_linalg::LADatum;
use tract_linalg::frame::mmm::FusedSpec;
use tract_linalg::frame::mmm::MatMatMulKer;

fn ruin_cache() {
    let _a = (0..1000000).collect::<Vec<i32>>();
}

fn bench_to_nanos<T: LADatum + Copy + num_traits::Zero, K: MatMatMulKer<T>>(
    k: usize,
    loops: usize,
) -> f64 {
    let item_size = T::datum_type().size_of();
    let a = Tensor::zero_aligned::<T>(
        &[(k + K::end_padding_packed_a()) * K::mr()],
        K::alignment_bytes_packed_a(),
    )
    .unwrap();
    let b = Tensor::zero_aligned::<T>(
        &[(k + K::end_padding_packed_b()) * K::nr()],
        K::alignment_bytes_packed_b(),
    )
    .unwrap();
    let mut c = Tensor::zero::<T>(&[K::mr() * K::nr()]).unwrap();
    let ref a = InputStoreKer::Packed { ptr: unsafe { a.as_ptr_unchecked::<u8>() as _ } };
    let ref b = InputStoreKer::Packed { ptr: unsafe { b.as_ptr_unchecked::<u8>() as _ } };
    let ref c = OutputStoreKer {
        ptr: unsafe { c.as_ptr_mut_unchecked::<u8>() as _ },
        item_size,
        col_byte_stride: (item_size * K::mr()) as isize,
        row_byte_stride: item_size as isize,
    };
    let ref linear = LinearSpec::Mul { k };
    let op = MatMatMulKerSpec { a, b, c, linear, non_linear: std::ptr::null() };
    let mut values = Vec::with_capacity(loops);
    for _ in 0..loops {
        ruin_cache();
        let start = Instant::now();
        K::kernel(&op);
        values.push(start.elapsed());
    }
    values.sort();
    values[loops / 2].as_nanos() as f64
}

fn model<T: Datum + Copy + num_traits::Zero, K: MatMatMulKer<T>>() -> (f64, f64) {
    let x = 1000;
    let zp = bench_to_nanos::<T, K>(0, 10000);
    let y = bench_to_nanos::<T, K>(x, 1000);
    let slope = (y - zp) / x as f64;
    (slope, zp)
}

fn as_match_line<T: Datum + Copy + num_traits::Zero, K: MatMatMulKer<T>>() {
    let coeffs = model::<T, K>();
    println!(
        "({:?}, {}, {}) => {} * k + {},",
        K::name(),
        K::mr(),
        K::nr(),
        (coeffs.0 * 1000.).round(),
        (coeffs.1 * 1000.).round()
    );
}

fn main() {
    use tract_linalg::arm64::*;
    as_match_line::<f32, MatMatMulF32x16x4>();
    as_match_line::<f32, MatMatMulF32x12x8>();
    as_match_line::<f32, MatMatMulF32x8x8>();
    as_match_line::<f32, MatMatMulF32x16x4A53>();
    as_match_line::<f32, MatMatMulF32x12x8A53>();
    as_match_line::<f32, MatMatMulF32x8x8A53>();
}
