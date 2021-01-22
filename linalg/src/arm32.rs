use std::{env, fs};
mod armv7neon;
mod armvfpv2;
use crate::frame::MatMatMulImpl;
use crate::frame::SigmoidImpl;
use crate::frame::TanhImpl;

use crate::Ops;

fn has_neon_cpuinfo() -> std::io::Result<bool> {
    let cpu_info = fs::read_to_string("/proc/cpuinfo")?;
    let neon = cpu_info.split("\n").any(|line| {
        line.starts_with("Features") && (line.contains("neon") || line.contains("asimd"))
    });
    Ok(neon)
}

fn has_neon() -> bool {
    if let Ok(v) = env::var("TRACT_CPU_ARM32_NEON") {
        return v == "true";
    }
    has_neon_cpuinfo().unwrap_or(false)
}

pub fn plug(ops: &mut Ops) {
    if has_neon() {
        log::info!("armv7neon activated (smmm, ssigmoid), stanh)");
        ops.mmm_f32 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x8x4, f32, f32, f32, f32>::new(m, k, n))
        });
        ops.qmmm_i8_i8 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<armv7neon::MatMatMulI8x8x4, i8, i8, i8, i32>::new(m, k, n))
        });
        ops.qmmm_i8_i32 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<armv7neon::MatMatMulI8xI32x8x4, i8, i8, i32, i32>::new(
                m, k, n,
            ))
        });
        ops.sigmoid_f32 =
            Box::new(|| Box::new(SigmoidImpl::<armv7neon::SigmoidF32x4n, f32>::new()));
        ops.tanh_f32 = Box::new(|| Box::new(TanhImpl::<armv7neon::TanhF32x4n, f32>::new()));
        ops.prefetch = Box::new(armv7neon::prefetch);
    } else {
        log::info!("armvfpv2 activated for smmm");
        ops.mmm_f32 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<armvfpv2::MatMatMulF32x4x4, f32, f32, f32, f32>::new(m, k, n))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn may_have_neon() {
        println!("Has neon ? {:?}", has_neon());
        if let Ok(neon) = env::var("TRACT_CPU_EXPECT_ARM32_NEON") {
            assert_eq!(neon == "true", has_neon());
        }
    }
}
