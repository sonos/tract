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

fn cpu_part() -> Option<usize> {
    fs::read_to_string("/proc/cpuinfo").ok().and_then(|cpuinfo| {
        cpuinfo
            .lines()
            .find(|line| line.starts_with("CPU part"))
            .and_then(|s| s.trim().split_whitespace().last())
            .and_then(|s| s.strip_prefix("0x"))
            .and_then(|s| usize::from_str_radix(s, 16).ok())
    })
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
        let cpu = cpu_part().unwrap_or(0);

        fn prefer_8x4(_m: Option<usize>, _k: Option<usize>, n:Option<usize>) -> bool {
            n.map(|n| n % 4 == 0 && n % 6 != 0 && n <= 12).unwrap_or(false)
        }

        ops.mmv_f32 = match cpu {
            0xc07 => Box::new(|_, _| {
                Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x32x1CortexA7, f32>::new())
            }),
            0xc09 => Box::new(|_, _| {
                Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x32x1CortexA9, f32>::new())
            }),
            _ => Box::new(|_, _| {
                Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x32x1Generic, f32>::new())
            }),
        };

        ops.mmm_f32 = match cpu {
            0xc07 => Box::new(|m, k, n| {
                if prefer_8x4(m, k, n) {
                    Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x8x4CortexA7, f32>::new())
                } else {
                    Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x8x6CortexA7, f32>::new())
                }
            }),
            0xc09 => Box::new(|m, k, n| {
                if prefer_8x4(m, k, n) {
                    Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x8x4CortexA9, f32>::new())
                } else {
                    Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x8x6CortexA9, f32>::new())
                }
            }),
            _ => Box::new(|m, k, n| {
                if prefer_8x4(m, k, n) {
                    Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x8x4Generic, f32>::new())
                } else {
                    Box::new(MatMatMulImpl::<armv7neon::MatMatMulF32x8x6Generic, f32>::new())
                }
            }),
        };
        ops.qmmm_i32 =
            Box::new(|_, _, _| Box::new(MatMatMulImpl::<armv7neon::MatMatMulI8x8x4, i32>::new()));
        ops.qmmv_i32 =
            Box::new(|_, _| Box::new(MatMatMulImpl::<armv7neon::MatMatMulI8x32x1, i32>::new()));
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
