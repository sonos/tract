//! Benchmark for the f32 max-reduction. Before the fix, ReduceMax ran the SIMD
//! `max_f32` kernel, threw the result away, then recomputed with a scalar fold;
//! after, it returns the SIMD result. Times `Reducer::Max` over a contiguous
//! trailing axis (the SIMD path).
//!
//! Run: cargo run --release --example reduce_max_bench -p tract-core
use std::time::Instant;

use tract_core::internal::*;
use tract_core::ops::nn::Reducer;

fn main() -> TractResult<()> {
    for (rows, k) in [(1024usize, 4096usize), (4096, 1024), (256, 65536)] {
        let n = rows * k;
        let data: Vec<f32> =
            (0..n).map(|i| (((i * 2654435761) >> 13) as f32 / 1e6).sin()).collect();
        let t = Tensor::from_shape(&[rows, k], &data)?;

        for _ in 0..3 {
            let _ = Reducer::Max.reduce(&[1], &t)?;
        }
        let runs = 50;
        let mut chk = 0f32;
        let s = Instant::now();
        for _ in 0..runs {
            let o = Reducer::Max.reduce(&[1], &t)?;
            chk += unsafe { o.as_slice_unchecked::<f32>() }[0];
            std::hint::black_box(&o);
        }
        let per = s.elapsed().as_secs_f64() / runs as f64;
        let gbps = (n * 4) as f64 / per / 1e9;
        println!(
            "reduce-max [{rows}x{k}] axis1 : {:>7.3} ms/call  {:>6.1} GB/s  (chk {chk:.3})",
            per * 1e3,
            gbps
        );
    }
    Ok(())
}
