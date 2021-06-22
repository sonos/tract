use std::time::{Duration, Instant};

use tract_data::prelude::*;
use tract_linalg::frame::mmm::LinearSpec;
use tract_linalg::frame::mmm::MatMatMulKer;
use tract_linalg::frame::mmm::MatMatMulKerSpec;
use tract_linalg::mmm::{PanelStore, Tile};

fn ruin_cache() {
    let _a = (0..1000000).collect::<Vec<i32>>();
}

fn bench<T: Datum + Copy + num_traits::Zero, K: MatMatMulKer<T>>(k: usize) -> Duration {
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
    let ref a = PanelStore::Packed { ptr: unsafe { a.as_ptr_unchecked::<u8>() as _ } };
    let ref b = PanelStore::Packed { ptr: unsafe { b.as_ptr_unchecked::<u8>() as _ } };
    let ref c = Tile {
        ptr: unsafe { c.as_ptr_mut_unchecked::<u8>() as _ },
        item_size,
        col_byte_stride: (item_size * K::mr()) as isize,
        row_byte_stride: item_size as isize,
    };
    let ref linear = LinearSpec::Mul { k };
    let op = MatMatMulKerSpec { a, b, c, linear, non_linear: std::ptr::null() };
    let mut duration = Duration::default();
    for _ in 0..1000 {
        ruin_cache();
        let start = Instant::now();
        K::kernel(&op);
        duration += start.elapsed()
    }
    duration
}

fn model<T: Datum + Copy + num_traits::Zero, K: MatMatMulKer<T>>() -> (f64, f64) {
    let x = 1000;
    let zp = bench::<T, K>(0).as_nanos() as f64;
    let y = bench::<T, K>(x).as_nanos() as f64;
    let slope = (y - zp) / x as f64;
    (slope, zp)
}

fn as_match_line<T: Datum + Copy + num_traits::Zero, K: MatMatMulKer<T>>() {
    let coeffs = model::<T,K>();
    println!("({:?}, {}, {}) => {} * k + {},", K::name(), K::mr(), K::nr(), coeffs.0.round(), coeffs.1.round());
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
