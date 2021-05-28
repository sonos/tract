mod arm64simd;

use crate::Ops;

use crate::frame::ElementWiseImpl;
use crate::frame::MatMatMulImpl;

use tract_data::internal::DimLike;

fn is_cortex_a53() -> std::io::Result<bool> {
    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")?;
    let a53 =
        cpu_info.split("\n").any(|line| line.starts_with("CPU part") && line.contains("0xd03"));
    Ok(a53)
}

pub fn plug(ops: &mut Ops) {
    if is_cortex_a53().unwrap_or(false) {
        log::info!("arm64simd activated for smmm (cortex A53)");
        ops.mmv_f32 =
            Box::new(|_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x64x1A53, f32>::new()));
        ops.mmm_f32 = Box::new(|m, _, _| {
            if m.is_some()
                && (m.unwrap() >= 128 || m.unwrap().div_ceil(12) * 12 <= m.unwrap().div_ceil(8) * 8)
            {
                Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x12x8A53, f32>::new())
            } else {
                Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8A53, f32>::new())
            }
        })
    } else {
        log::info!("arm64simd activated for smmm (generic)");
        ops.mmv_f32 =
            Box::new(|_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x64x1, f32>::new()));
        ops.mmm_f32 = Box::new(|m, _, _| {
            if m.is_some()
                && (m.unwrap() >= 128 || m.unwrap().div_ceil(12) * 12 <= m.unwrap().div_ceil(8) * 8)
            {
                Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x12x8, f32>::new())
            } else {
                Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8, f32>::new())
            }
        })
    }

    ops.qmmm_i8_i8 =
        Box::new(|_, _, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8x8x8, i32>::new()));
    ops.qmmv_i8_i8 =
        Box::new(|_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8x64x1, i32>::new()));
    ops.qmmm_i8_i32 =
        Box::new(|_, _, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8xI32x8x8, i32>::new()));
    ops.qmmv_i8_i32 =
        Box::new(|_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8xI32x64x1, i32>::new()));

    ops.sigmoid_f32 =
        Box::new(|| Box::new(ElementWiseImpl::<arm64simd::SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<arm64simd::TanhF32x4n, f32>::new()));
}
