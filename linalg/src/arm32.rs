use std::{env, fs};
mod armv7neon;
mod armvfpv2;
mod cortex_a7;
mod cortex_a9;

use crate::frame::mmm::kernel::MatMatMulKer;
use crate::frame::ElementWiseImpl;

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

        ops.mmv_f32 = match cpu {
            0xc07 => Box::new(|_, _| armv7neon::MatMatMulF32x32x1CortexA7::mmm()),
            0xc09 => Box::new(|_, _| armv7neon::MatMatMulF32x32x1CortexA9::mmm()),
            _ => Box::new(|_, _| armv7neon::MatMatMulF32x32x1Generic::mmm()),
        };

        for k in vec![
            armv7neon::MatMatMulF32x8x4CortexA7::mmm(),
            armv7neon::MatMatMulF32x8x6CortexA7::mmm(),
            armv7neon::MatMatMulF32x8x4CortexA9::mmm(),
            armv7neon::MatMatMulF32x8x6CortexA9::mmm(),
            armv7neon::MatMatMulF32x8x4Generic::mmm(),
            armv7neon::MatMatMulF32x8x6Generic::mmm(),
        ] {
            ops.mmm_f32_impls.push((k, None));
        }

        ops.qmmm_i32 = Box::new(|_, _, _| armv7neon::MatMatMulI32x8x4::mmm());
        ops.qmmv_i32 = Box::new(|_, _| armv7neon::MatMatMulI32x32x1::mmm());
        ops.sigmoid_f32 =
            Box::new(|| Box::new(ElementWiseImpl::<armv7neon::SigmoidF32x4n, f32>::new()));
        ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<armv7neon::TanhF32x4n, f32>::new()));

        match cpu {
            0xc07 => ops.set_cost_models(cortex_a7::models()),
            0xc09 => ops.set_cost_models(cortex_a9::models()),
            _ => ops.set_cost_models(cortex_a7::models()),
        };
    } else {
        log::info!("armvfpv2 activated for smmm");
        ops.mmm_f32_impls.push((armvfpv2::MatMatMulF32x4x4::mmm(), None));
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
