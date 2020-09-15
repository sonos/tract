mod arm64simd;

use crate::Ops;

use crate::frame::MatMatMulImpl;
use crate::frame::QMatMatMulImpl;
use crate::frame::SigmoidImpl;
use crate::frame::TanhImpl;

fn is_a7x() -> std::io::Result<bool> {
    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")?;
    let neon =
        cpu_info.split("\n").any(|line| line.starts_with("CPU part") && line.contains("0xd08"));
    Ok(neon)
}

pub fn plug(ops: &mut Ops) {
    let is_a7x = is_a7x().unwrap_or(true);
    log::info!("arm64simd activated for smmm (A7x: {:?})", is_a7x);
    ops.mmm_f32 = Box::new(move |m, k, n| {
        if is_a7x {
            Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8A7x, f32, f32, f32, f32>::new(
                m, k, n,
            ))
        } else {
            Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8A5x, f32, f32, f32, f32>::new(
                m, k, n,
            ))
        }
    });
    ops.qmmm_i8_i8 = Box::new(|m, k, n| {
        Box::new(QMatMatMulImpl::from(
            MatMatMulImpl::<arm64simd::MatMatMulI8x8x8, i8, i8, i8, i32>::new(m, k, n),
        ))
    });
    ops.qmmm_i8_i32 = Box::new(|m, k, n| {
        Box::new(QMatMatMulImpl::from(MatMatMulImpl::<
            arm64simd::MatMatMulI8xI32x8x8,
            i8,
            i8,
            i32,
            i32,
        >::new(m, k, n)))
    });
    ops.sigmoid_f32 = Box::new(|| Box::new(SigmoidImpl::<arm64simd::SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(TanhImpl::<arm64simd::TanhF32x4n, f32>::new()));
}
