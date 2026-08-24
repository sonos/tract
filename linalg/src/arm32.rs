use std::fs;
pub mod armv7neon;
mod armvfpv2;
mod cortex_a7_linear;
mod cortex_a7_mmv_linear;
mod cortex_a9_linear;
mod cortex_a9_mmv_linear;
use crate::DatumType;
use crate::isa::IsaSet;
use crate::mmm::{Query, Suitable};

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

fn prefer_8x4(n: Option<usize>) -> bool {
    n.map(|n| n % 4 == 0 && n % 6 != 0 && n <= 12).unwrap_or(false)
}

/// Consulted only for small m (see arm64 for the rationale).
fn neon_mmv_f32(suitable: &[Suitable], query: &Query) -> Option<&'static str> {
    match cpu_part().unwrap_or(0) {
        0xc07 => match query.m {
            Some(m) if m < 32 => {
                cortex_a7_mmv_linear::linear_model().preferred(suitable, Some(m), query.k, Some(1))
            }
            _ => Some(&armv7neon::armv7neon_mmm_f32_32x1_cortexa7.name.as_str()),
        },
        0xc09 => match query.m {
            Some(m) if m < 32 => {
                cortex_a9_mmv_linear::linear_model().preferred(suitable, Some(m), query.k, Some(1))
            }
            _ => Some(&armv7neon::armv7neon_mmm_f32_32x1_cortexa9.name.as_str()),
        },
        _ => Some(&armv7neon::armv7neon_mmm_f32_32x1_generic.name.as_str()),
    }
}

fn neon_mmm_f32(suitable: &[Suitable], query: &Query) -> Option<&'static str> {
    match cpu_part().unwrap_or(0) {
        0xc07 => cortex_a7_linear::linear_model().preferred(suitable, query.m, query.k, query.n),
        0xc09 => cortex_a9_linear::linear_model().preferred(suitable, query.m, query.k, query.n),
        _ => Some(if prefer_8x4(query.n) {
            &armv7neon::armv7neon_mmm_f32_8x4_generic.name
        } else {
            &armv7neon::armv7neon_mmm_f32_8x6_generic.name
        }),
    }
}

fn preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::F32, Some(1)) => neon_mmv_f32(suitable, query),
        (DatumType::F32, _) => neon_mmm_f32(suitable, query),
        (DatumType::I32, Some(1)) => Some(&armv7neon::armv7neon_mmm_i32_32x1.name.as_str()),
        (DatumType::I32, _) => Some(&armv7neon::armv7neon_mmm_i32_8x4.name.as_str()),
        _ => None,
    }
}

inventory::submit! {
    crate::mmm_tiers::MmmTier {
        arch: Some(crate::isa::Arch::Arm),
        precedence: 2,
        name: "armv7neon",
        applies: |isa| isa.has(crate::isa::Isa::ArmNeon),
        preferred,
    }
}

/// What this core has, in the shared vocabulary.
pub fn isa_set() -> crate::isa::IsaSet {
    use crate::isa::{Isa, IsaSet};
    let set = IsaSet::of_arch(crate::isa::Arch::Arm);
    if has_neon() { set.with(Isa::ArmNeon) } else { set }
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
