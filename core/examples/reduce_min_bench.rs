//! Benchmark for the f32 min-reduction. `min_t` now routes the contiguous f32
//! case through the new SIMD `min_f32` (generic `SMin4`) reducer instead of the
//! scalar branchy fold. Times `Reducer::Min` (SIMD) against an inline replica of
//! the previous scalar fold over the same data.
//!
//! Run: cargo run --release --example reduce_min_bench -p tract-core
use std::time::Instant;

use tract_core::internal::*;
use tract_core::ops::nn::Reducer;

#[inline(never)]
fn scalar_min_per_row(data: &[f32], rows: usize, k: usize) -> f32 {
    // replica of the old min_t scalar fold (branchy partial-ord)
    let mut acc = 0f32;
    for r in 0..rows {
        let m = data[r * k..(r + 1) * k]
            .iter()
            .copied()
            .fold(f32::MAX, |a, v| if a < v { a } else { v });
        acc += m;
    }
    acc
}

fn main() -> TractResult<()> {
    for (rows, k) in [(1024usize, 4096usize), (4096, 1024), (256, 65536)] {
        let n = rows * k;
        let data: Vec<f32> =
            (0..n).map(|i| (((i * 2654435761) >> 13) as f32 / 1e6).sin()).collect();
        let t = Tensor::from_shape(&[rows, k], &data)?;

        // SIMD path (Reducer::Min)
        for _ in 0..3 {
            let _ = Reducer::Min.reduce(&[1], &t)?;
        }
        let runs = 50;
        let s = Instant::now();
        for _ in 0..runs {
            std::hint::black_box(Reducer::Min.reduce(&[1], &t)?);
        }
        let simd = s.elapsed().as_secs_f64() / runs as f64;

        // scalar replica of the old fold
        for _ in 0..3 {
            std::hint::black_box(scalar_min_per_row(&data, rows, k));
        }
        let s = Instant::now();
        for _ in 0..runs {
            std::hint::black_box(scalar_min_per_row(&data, rows, k));
        }
        let scalar = s.elapsed().as_secs_f64() / runs as f64;

        println!(
            "min [{rows}x{k}] axis1 : scalar {:>7.3} ms -> SIMD {:>7.3} ms  ({:.1}x, {:>5.1} GB/s)",
            scalar * 1e3,
            simd * 1e3,
            scalar / simd,
            (n * 4) as f64 / simd / 1e9
        );
    }
    Ok(())
}
