use std::{env, fs};
mod armv7neon;
mod armvfpv2;
use crate::frame::ElementWiseImpl;
use crate::frame::MatMatMulImpl;

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
        return v == "true" || v == "1";
    }
    has_neon_cpuinfo().unwrap_or(false)
}

pub fn plug(ops: &mut Ops) {
    if has_neon() {
        log::info!("armv7neon activated (smmm, ssigmoid), stanh)");
        ops.mmv_f32 =
            Box::new(|_, _| Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x32x1, f32>::new()));
        ops.mmm_f32 =
            Box::new(|_, _, _| Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x8x4, f32>::new()));
        ops.qmmm_i8_i8 =
            Box::new(|_, _, _| Box::new(MatMatMulImpl::<armv7neon::MatMatMulI8x8x4, i32>::new()));
        ops.qmmv_i8_i8 =
            Box::new(|_, _| Box::new(MatMatMulImpl::<armv7neon::MatMatMulI8x32x1, i8, i32>::new()));
        ops.qmmm_i8_i32 = Box::new(|_, _, _| {
            Box::new(MatMatMulImpl::<armv7neon::MatMatMulI8xI32x8x4, i32>::new())
        });
        ops.qmmv_i8_i32 = Box::new(|_, _| {
            Box::new(MatMatMulImpl::<armv7neon::MatMatMulI8xI32x32x1, i32, i32>::new())
        });
        ops.sigmoid_f32 =
            Box::new(|| Box::new(ElementWiseImpl::<armv7neon::SigmoidF32x4n, f32>::new()));
        ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<armv7neon::TanhF32x4n, f32>::new()));
        ops.prefetch = Some(&armv7neon::prefetch);
    } else {
        log::info!("armvfpv2 activated for smmm");
        ops.mmm_f32 =
            Box::new(|_, _, _| Box::new(MatMatMulImpl::<armvfpv2::MatMatMulF32x4x4, f32>::new()));
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
