//! Micro-benchmark for Metal element-wise kernel threadgroup occupancy.
//!
//! Times the converted flat (`thread_position_in_grid`) kernels — silu (the FFN
//! activation, compute-bound via exp) and cast (memory-bound) — over large
//! tensors. Compare before/after the `dispatch_threads_1d` change: the old code
//! dispatched `n` threadgroups of a single thread, leaving 31/32 SIMD lanes idle.
//!
//! Run: cargo run --release --example threadgroup_bench -p tract-metal
use std::time::Instant;

use tract_core::internal::*;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};
use tract_metal::kernels::array::Cast;
use tract_metal::kernels::nn::Silu;
use tract_metal::with_metal_stream;

fn main() -> TractResult<()> {
    with_metal_stream(|stream| {
        let k = 200;
        println!("{:>6} | {:>12} | {:>11} | {:>10}", "op", "n_elements", "us/call", "GB/s");
        println!("{:-<6}-+-{:-<12}-+-{:-<11}-+-{:-<10}", "", "", "", "");
        for &n in &[1usize << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24] {
            // ---- silu f32 (in == out dt) ----
            let input = Tensor::from_shape(&[n], &vec![0.5f32; n])?.into_device()?;
            let out = DeviceTensor::uninitialized_dt(DatumType::F32, &[n])?;
            for _ in 0..5 {
                Silu.dispatch_eval(stream, &input, &out)?;
            }
            stream.wait_until_completed()?;
            let t = Instant::now();
            for _ in 0..k {
                Silu.dispatch_eval(stream, &input, &out)?;
            }
            stream.wait_until_completed()?;
            let us = t.elapsed().as_secs_f64() * 1e6 / k as f64;
            let gbps = (n as f64 * 4.0 * 2.0) / (us * 1e-6) / 1e9;
            println!("{:>6} | {:>12} | {:>11.2} | {:>10.1}", "silu", n, us, gbps);

            // ---- cast f32 -> f16 ----
            let out16 = DeviceTensor::uninitialized_dt(DatumType::F16, &[n])?;
            for _ in 0..5 {
                Cast.dispatch_eval(stream, &input, &out16)?;
            }
            stream.wait_until_completed()?;
            let t = Instant::now();
            for _ in 0..k {
                Cast.dispatch_eval(stream, &input, &out16)?;
            }
            stream.wait_until_completed()?;
            let us = t.elapsed().as_secs_f64() * 1e6 / k as f64;
            let gbps = (n as f64 * (4.0 + 2.0)) / (us * 1e-6) / 1e9;
            println!("{:>6} | {:>12} | {:>11.2} | {:>10.1}", "cast", n, us, gbps);
        }
        Ok(())
    })
}
