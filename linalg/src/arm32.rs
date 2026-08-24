use std::fs;
pub mod armv7neon;
mod armvfpv2;
mod cortex_a7_linear;
mod cortex_a7_mmv_linear;
mod cortex_a9_linear;
mod cortex_a9_mmv_linear;
use armv7neon::*;

use crate::frame::element_wise::ElementWiseKer;

use crate::mmm::candidate_named;
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

        // Consulted only for small m (see arm64 for the rationale).
        let mmv_f32: crate::MmmPreference = match cpu {
            0xc07 => {
                let model = cortex_a7_mmv_linear::linear_model();
                Box::new(move |candidates, query| match query.m {
                    Some(m) if m < 32 => model.preferred(candidates, Some(m), query.k, Some(1)),
                    _ => candidate_named(
                        candidates,
                        &armv7neon::armv7neon_mmm_f32_32x1_cortexa7.name,
                    ),
                })
            }
            0xc09 => {
                let model = cortex_a9_mmv_linear::linear_model();
                Box::new(move |candidates, query| match query.m {
                    Some(m) if m < 32 => model.preferred(candidates, Some(m), query.k, Some(1)),
                    _ => candidate_named(
                        candidates,
                        &armv7neon::armv7neon_mmm_f32_32x1_cortexa9.name,
                    ),
                })
            }
            _ => Box::new(|candidates, _| {
                candidate_named(candidates, &armv7neon::armv7neon_mmm_f32_32x1_generic.name)
            }),
        };

        let mmm_f32: crate::MmmPreference = match cpu {
            0xc07 => {
                let model = cortex_a7_linear::linear_model();
                Box::new(move |candidates, query| {
                    model.preferred(candidates, query.m, query.k, query.n)
                })
            }
            0xc09 => {
                let model = cortex_a9_linear::linear_model();
                Box::new(move |candidates, query| {
                    model.preferred(candidates, query.m, query.k, query.n)
                })
            }
            _ => Box::new(|candidates, query| {
                candidate_named(
                    candidates,
                    if prefer_8x4(query.m, query.k, query.n) {
                        &armv7neon::armv7neon_mmm_f32_8x4_generic.name
                    } else {
                        &armv7neon::armv7neon_mmm_f32_8x6_generic.name
                    },
                )
            }),
        };
        ops.overlay_mmm_policy(move |prev, dt, query, candidates| match (dt, query.n) {
            (DatumType::F32, Some(1)) => mmv_f32(candidates, query),
            (DatumType::F32, _) => mmm_f32(candidates, query),
            (DatumType::I32, Some(1)) => {
                candidate_named(candidates, &armv7neon::armv7neon_mmm_i32_32x1.name)
            }
            (DatumType::I32, _) => {
                candidate_named(candidates, &armv7neon::armv7neon_mmm_i32_8x4.name)
            }
            _ => prev(dt, query, candidates),
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
