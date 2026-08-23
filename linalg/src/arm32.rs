use std::fs;
pub mod armv7neon;
mod armvfpv2;
mod cortex_a7_linear;
mod cortex_a7_mmv_linear;
mod cortex_a9_linear;
mod cortex_a9_mmv_linear;
use armv7neon::*;

use crate::frame::element_wise::ElementWiseKer;

use crate::{DatumType, Ops};

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
            .and_then(|s| s.split_whitespace().last())
            .and_then(|s| s.strip_prefix("0x"))
            .and_then(|s| usize::from_str_radix(s, 16).ok())
    })
}

pub(crate) fn has_neon() -> bool {
    if let Some(forced) = crate::knobs::TRACT_CPU_ARM32_NEON.get() {
        return forced;
    }
    static NEON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *NEON.get_or_init(|| has_neon_cpuinfo().unwrap_or(false))
}

pub fn plug(ops: &mut Ops) {
    if crate::isa::native().has(crate::isa::Isa::Neon) {
        log::info!("armv7neon activated (smmm, ssigmoid), stanh)");

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
        // The mmv model picks over the full pool (it needs the nr==1 matvec kernels, which the
        // mmm cost_managed pool omits); consulted only for small m (see arm64 for the rationale).
        let mmv_impls = ops.mmm_impls.clone();
        let mmv_f32: crate::MMMImpl = match cpu {
            0xc07 => {
                let model = cortex_a7_mmv_linear::linear_model();
                let impls = mmv_impls.clone();
                Box::new(move |m, k, _| match m {
                    Some(m) if m < 32 => model.pick(&impls, Some(m), k, Some(1)),
                    _ => armv7neon::armv7neon_mmm_f32_32x1_cortexa7.mmm(),
                })
            }
            0xc09 => {
                let model = cortex_a9_mmv_linear::linear_model();
                let impls = mmv_impls.clone();
                Box::new(move |m, k, _| match m {
                    Some(m) if m < 32 => model.pick(&impls, Some(m), k, Some(1)),
                    _ => armv7neon::armv7neon_mmm_f32_32x1_cortexa9.mmm(),
                })
            }
            _ => Box::new(|_, _, _| armv7neon::armv7neon_mmm_f32_32x1_generic.mmm()),
        };

        let mmm_f32: crate::MMMImpl = match cpu {
            0xc07 => {
                let model = cortex_a7_linear::linear_model();
                Box::new(move |m, k, n| model.pick(&cost_managed_impls, m, k, n))
            }
            0xc09 => {
                let model = cortex_a9_linear::linear_model();
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
        ops.overlay_mmm_policy(move |prev, dt, m, k, n| match (dt, n) {
            (DatumType::F32, Some(1)) => Some(mmv_f32(m, k, n)),
            (DatumType::F32, _) => Some(mmm_f32(m, k, n)),
            (DatumType::I32, Some(1)) => Some(armv7neon::armv7neon_mmm_i32_32x1.mmm()),
            (DatumType::I32, _) => Some(armv7neon::armv7neon_mmm_i32_8x4.mmm()),
            _ => prev(dt, m, k, n),
        });
        ops.sigmoid_f32 = Box::new(|| armv7neon_sigmoid_f32_4n::ew());
        ops.silu_f32 = Box::new(|| armv7neon_silu_f32_4n::ew());
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
        if let Ok(neon) = std::env::var("TRACT_CPU_EXPECT_ARM32_NEON") {
            assert_eq!(neon == "true", has_neon());
        }
    }
}

inventory::submit! {
    crate::platform::PlatformSelector {
        target: crate::platform::Target::Arm,
        plug,
    }
}

/// What this core has, in the shared vocabulary.
pub fn isa_set() -> crate::isa::IsaSet {
    use crate::isa::{Isa, IsaSet};
    let set = IsaSet::empty();
    if has_neon() { set.with(Isa::Neon) } else { set }
}
