mod arm64simd;

use crate::Ops;

use crate::frame::MatMatMulImpl;
use crate::frame::SigmoidImpl;
use crate::frame::TanhImpl;

fn is_cortex_a5x() -> std::io::Result<bool> {
    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")?;
    let a5x = cpu_info.split("\n").any(|line| {
        line.starts_with("CPU part") && ["0xd03", "0xd07"].iter().any(|s| line.contains(s))
    });
    Ok(a5x)
}

pub fn plug(ops: &mut Ops) {
    if is_cortex_a5x().unwrap_or(false) {
        log::info!("arm64simd activated for smmm (cortex A53/A55 variant)");
        ops.mmm_f32 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8A5x, f32, f32>::new(m, k, n))
        });
    } else {
        log::info!("arm64simd activated for smmm (generic variant)");
        ops.mmm_f32 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8, f32, f32>::new(m, k, n))
        })
    }
    ops.qmmm_i8_i8 = Box::new(|m, k, n| {
        Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8x8x8, i8, i32>::new(m, k, n))
    });
    ops.qmmm_i8_i32 = Box::new(|m, k, n| {
        Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8xI32x8x8, i32, i32>::new(m, k, n))
    });
    ops.sigmoid_f32 = Box::new(|| Box::new(SigmoidImpl::<arm64simd::SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(TanhImpl::<arm64simd::TanhF32x4n, f32>::new()));
}
