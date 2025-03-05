use std::{env, fs};
pub mod armv7neon;
mod armvfpv2;
mod cortex_a7;
mod cortex_a9;
use armv7neon::*;

use crate::frame::element_wise::ElementWiseKer;

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
        armv7neon::plug(ops);

        let cpu = cpu_part().unwrap_or(0);

        fn prefer_8x4(_m: Option<usize>, _k: Option<usize>, n: Option<usize>) -> bool {
            n.map(|n| n % 4 == 0 && n % 6 != 0 && n <= 12).unwrap_or(false)
        }

        let cost_managed_impls = vec![
            armv7neon_mmm_f32_8x4_cortexa7.mmm(),
            armv7neon_mmm_f32_8x6_cortexa7.mmm(),
            armv7neon_mmm_f32_8x4_cortexa9.mmm(),
            armv7neon_mmm_f32_8x6_cortexa9.mmm(),
            armv7neon_mmm_f32_8x4_generic.mmm(),
            armv7neon_mmm_f32_8x6_generic.mmm(),
            crate::generic::mmm::generic_f32_4x4.mmm(),
        ];
        ops.mmv_f32 = match cpu {
            0xc07 => Box::new(|_, _| armv7neon::armv7neon_mmm_f32_32x1_cortexa7.mmm()),
            0xc09 => Box::new(|_, _| armv7neon::armv7neon_mmm_f32_32x1_cortexa9.mmm()),
            _ => Box::new(|_, _| armv7neon::armv7neon_mmm_f32_32x1_generic.mmm()),
        };

        ops.mmm_f32 = match cpu {
            0xc07 => {
                let model = cortex_a7::model();
                Box::new(move |m, k, n| model.pick(&cost_managed_impls, m, k, n))
            }
            0xc09 => {
                let model = cortex_a9::model();
                Box::new(move |m, k, n| model.pick(&cost_managed_impls, m, k, n))
            }
            _ => Box::new(|m, k, n| {
                if prefer_8x4(m, k, n) {
                    armv7neon::armv7neon_mmm_f32_8x4_generic.mmm()
                } else {
                    armv7neon::armv7neon_mmm_f32_8x6_generic.mmm()
                }
            }),
        };
        ops.qmmm_i32 = Box::new(|_, _, _| armv7neon::armv7neon_mmm_i32_8x4.mmm());
        ops.qmmv_i32 = Box::new(|_, _| armv7neon::armv7neon_mmm_i32_32x1.mmm());
        ops.sigmoid_f32 = Box::new(|| armv7neon_sigmoid_f32_4n::ew());
        ops.tanh_f32 = Box::new(|| armv7neon_tanh_f32_4n::ew());
    } else {
        armvfpv2::plug(ops);
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
